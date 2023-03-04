#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os, sys, copy
import time, random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from models.resnet import ResNetWrapper
from models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from options import args_parser
from models.multibackbone import alexnet, vgg11, mlp_m
from data_utils import DatasetSplit
from utils import average_weights

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion_CE = nn.CrossEntropyLoss().to(args.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, model, global_round, global_fcs):
        # Set mode to train model
        model_dg = copy.deepcopy(model)
        model_dg.eval()
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()                                
                logits, feature = model(images, get_features=True)

                model_dg = copy.deepcopy(model_dg).to('cuda')
                loss_ce_temp = 0. 
                for i, fc in enumerate(global_fcs):
                    if i==idx: continue
                    model_dg.state_dict()['fc'].data.copy_(fc)
                    logits_temp = model_dg.fc(feature)
                    loss_ce_temp += self.criterion_CE(logits, labels)
                loss_ce_temp /= len(global_fcs) - 1

                loss_ce = self.criterion_CE(logits, labels)
                loss = loss_ce + loss_ce_temp 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(args, dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss().to(args.device)

    def test_split(self, args, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=args.test_bs, shuffle=False)
        return testloader

    def test_inference(self, args, model):
        device = args.device
        model.eval().to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0

        # test (only use local model)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss += self.criterion(logits, labels)
            _, pred_labels = torch.max(logits, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        loss /= (batch_idx + 1)

        return acc, loss

def FedAvg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, local_model_list):
    train_loss, train_accuracy = [], []
    global_model = ResNetWrapper(local_model_list, args.num_classes)
    global_model.to('cuda')

    global_fcs = []
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round} |\n')
        print(datetime.now())
        
        for idx in range(args.num_users):
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights(idx, model=copy.deepcopy(global_model), global_round=round, global_fcs=global_fcs)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)

        # update global weights
        local_weights_list = average_weights(local_weights)        
        global_model.load_state_dict(local_weights_list[0], strict=True)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        if round % 10 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    acc, loss = local_test.test_inference(args, global_model)
                    print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)

    acc_mtx = torch.zeros([args.num_users])
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            acc, loss = local_test.test_inference(args, global_model)
            print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss

    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))

    return acc_mtx