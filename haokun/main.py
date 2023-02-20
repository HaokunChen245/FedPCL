#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os, sys, copy
import time, random
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime
from fedavg import FedAvg

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


from models.resnet import resnet18
from models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from options import args_parser
from update import LocalUpdate, LocalTest
from models.multibackbone import alexnet, vgg11, mlp_m
from utils import (
    prepare_data_real_noniid, 
    prepare_data_domainnet_noniid, 
    prepare_data_office_noniid,
    prepare_data_digits_noniid, 
    prepare_data_caltech_noniid, 
    prepare_data_mnistm_noniid, 
    average_weights, 
    exp_details, 
    prepare_data_digits, 
    prepare_data_office, 
    prepare_data_domainnet,
    _random_seeder
)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def main(args):
    exp_details(args)

    # set random seed
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    print("Training on", args.device, '...')
    _random_seeder(args.seed)

    # dataset initialization
    # feature iid, label non-iid
    if args.feature_iid and args.label_iid==0:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_mnistm_noniid(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_caltech_noniid(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_real_noniid(args.num_users, args=args)
    # feature non-iid, label iid
    elif args.feature_iid==0 and args.label_iid:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet(args.num_users, args=args)
    # feature non-iid, label non-iid
    elif args.feature_iid==0 and args.label_iid==0:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits_noniid(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office_noniid(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet_noniid(args.num_users, args=args)

    # load backbone
    if args.model == 'cnn':
        resnet_quickdraw = resnet18(pretrained=True, ds='quickdraw', data_dir=args.data_dir)
        resnet_birds = resnet18(pretrained=True, ds='birds', data_dir=args.data_dir)
        resnet_aircraft = resnet18(pretrained=True, ds='aircraft', data_dir=args.data_dir)
    elif args.model == 'vit':
        vit_t = vit_tiny_patch16_224(pretrained=False)
        vit_t.load_pretrained(args.data_dir + 'weights/Ti_16-i1k-300ep-lr_0.001-aug_light0-wd_0.1-do_0.0-sd_0.0.npz')
        vit_s = vit_small_patch16_224(pretrained=False)
        vit_s.load_pretrained(args.data_dir + 'weights/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz')
        vit_b = vit_base_patch16_224(pretrained=False)
        vit_b.load_pretrained(args.data_dir + 'weights/B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1.npz')

    # model initialization
    local_model_list = []
    for _ in range(args.num_users):
        if args.num_bb == 1:
            if args.model == 'cnn':
                models = resnet_quickdraw
            elif args.model == 'vit':
                models = vit_s
        elif args.num_bb == 3:
            if args.model == 'cnn':
                backbone_list = [resnet_quickdraw, resnet_aircraft, resnet_birds]
                local_model = ProjandDeci(512*3, 256, 10)
            elif args.model == 'vit':
                backbone_list = [vit_t, vit_s, vit_b]
                local_model = ProjandDeci(192+384+768, 256, 10)
            elif args.model == 'other':
                MLP=mlp_m(pretrained=True)
                AlexNet=alexnet(pretrained=True)
                VGG=vgg11(pretrained=True)
                backbone_list = [MLP, AlexNet, VGG]
                local_model = ProjandDeci(4352, 256, 10)
        local_model_list.append(models)

    print(args)
    summary_writer = SummaryWriter('./tensorboard/' + args.dataset + '_' + args.alg + '_' + str(len(models)) + 'bb_' + str(args.rounds) + 'r_' + str(args.num_users) + 'u_'+ str(args.train_ep) + 'ep')
    if args.alg == 'fedavg':
        acc_mtx = FedAvg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, local_model_list)
    elif args.alg == 'local':
        acc_mtx = Local(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, local_model_list)

    return acc_mtx

if __name__ == '__main__':
    num_trial = 3
    args = args_parser()
    acc_mtx = torch.zeros([num_trial, args.num_users])

    for i in range(num_trial):
        args.seed = i
        acc_mtx[i,:] = main(args)

    print("The avg test acc of all trials are:")
    for j in range(args.num_users):
        print('{:.2f}'.format(torch.mean(acc_mtx[:,j])*100))

    print("The stdev of test acc of all trials are:")
    for j in range(args.num_users):
        print('{:.2f}'.format(torch.std(acc_mtx[:,j])*100))

    acc_avg = torch.zeros([num_trial])
    for i in range(num_trial):
        acc_avg[i] = torch.mean(acc_mtx[i,:]) * 100
    print("The avg and stdev test acc of all clients in the trials:")
    print('{:.2f}'.format(torch.mean(acc_avg)))
    print('{:.2f}'.format(torch.std(acc_avg)))