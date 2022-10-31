'''
Author: ssy
Date: 2022-10-19 22:14:17
LastEditors: ssy
LastEditTime: 2022-10-20 02:54:22
FilePath: /SCFMTL/experiment.py
Description: 实现隐私保护的SCFMTL方案，

Copyright (c) 2022 by ssy, All Rights Reserved. 
'''

import os

import torch

from CFMTL.data import *
from CFMTL.local import Local_Update3
from CFMTL.test import Test
from CFMTL.model import *

from aggre import *
import numpy as np

from crypten.mpc import multiprocess_wrap

from tqdm import tqdm

def experiment_caltech101(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    acc_train = []
    groups = [[i for i in range(args.num_clients)]]
    loss_train = []

    w_groups = [Net.state_dict()]

    for iter in tqdm(range(args.ep)):
        loss_local = []
        num_group = len(groups)

        w_local = [None for i in range(args.num_clients)]  # 所有客户端w的集合torch.save

        # 并行在客户端执行的
        print("Begin trainning locally")
        for group_id in range(num_group):
            group = groups[group_id]
            if iter > 0:
                num_clients = max(int(args.frac * len(group)), 1)
                clients = np.random.choice(group, num_clients, replace=False)
                group = clients
            w_group = w_groups[group_id]
            for id in group:
                mem, w, loss = Local_Update3(args, w_group, dataset_train, dict_train[id], iter)
                w_local[id] = w  # 添加到server端
                loss_local.append(loss)
        print("End trainning locally...")
        print("Begin to cluster ... ")
        if iter == 0:
            torch.save(w_local, './w_local.pth')
            del w_local
            torch.cuda.empty_cache()
            print("save w_local success")
            multiprocess_wrap(Cluster_Init, world_size=2, args=(args,))
            groups, w_groups, _, _ = torch.load("./rank0.pth")

        else:
            torch.save(w_local, './w_local_sub.pth')

            multiprocess_wrap(Cluster_FedAvg, world_size=2, args=(args,))
            w_groups = torch.load("./w_groups.pth")
        print("End to cluster ... ")
        loss_avg = sum(loss_local) / len(loss_local)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter == 0:
            print("Groups Number: ", len(groups))
            print(groups)

        acc_test = []
        num_group = len(groups)
        for group_id in range(num_group):
            group = groups[group_id]
            w_group = w_groups[group_id]
            for id in group:
                acc, loss_test = Test(args, w_group, dataset_test, dict_test[id])
                acc_test.append(acc)
        acc_avg = sum(acc_test) / len(acc_test)
        print("Testing accuracy: {:.2f}".format(acc_avg))
        acc_train.append(acc_avg.item())


    return acc_train

    
    
