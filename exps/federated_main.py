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

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


from models.resnet import resnet18
from models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from options import args_parser
from update import LocalUpdate, LocalTest
from models.models import ProjandDeci
from models.multibackbone import alexnet, vgg11, mlp_m
from utils import add_noise_proto, prepare_data_real_noniid, prepare_data_domainnet_noniid, prepare_data_office_noniid, prepare_data_digits_noniid, prepare_data_caltech_noniid, prepare_data_mnistm_noniid, average_weights, exp_details, proto_aggregation, agg_func, prepare_data_digits, prepare_data_office, prepare_data_domainnet



def FedPCL(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    global_protos = {}
    global_avg_protos = {}
    local_protos = {}

    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list[idx], idxs=user_groups[idx])
            w, w_urt, loss, protos = local_model.update_weights_lg(args, idx, global_protos, global_avg_protos, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            if args.add_noise_proto:
                agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.perturb_coe, args.noise_type)

            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_protos[idx] = copy.deepcopy(agg_protos)

            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx), loss['2'], round)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights[idx])

        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        summary_writer.add_scalar('Train/Loss/avg', loss_avg, round)

        if round % 20 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)

    acc_mtx = torch.zeros([args.num_users])
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list[idx], idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss

    return acc_mtx

