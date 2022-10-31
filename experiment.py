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
from CFMTL.local import Local_Update3, Local_Update
from CFMTL.test import Test
from CFMTL.model import *

from aggre import *
import numpy as np

from crypten.mpc import multiprocess_wrap

from tqdm import tqdm
import multiprocessing as mp

def save_result(args, acc_final, pro_final, m):
    if args.iid == "iid":
        filename = f'experiments/{args.experiment}-iid-secure.npy'
    elif args.iid == "non-iid":
        filename = f'experiments/{args.experiment}-noniid-{args.ratio}-secure.npy'
    else:  # non-iid-single_class
        filename = f'experiments/{args.experiment}-non-iid-single_class-secure.npy'

    if os.path.exists(filename):
        record = np.load(filename, allow_pickle=True).tolist()
        record[0][m] = copy.deepcopy(acc_final[m])
        record[1][m] = copy.deepcopy(pro_final[m])
    else:
        record = [copy.deepcopy(acc_final), copy.deepcopy(pro_final)]
    record = np.array(record)
    np.save(filename, record)

def experiment_mnist(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    """在mnist数据集上的密文实现方案
    """
    acc_final = [[] for i in range(3)]
    pro_final = [[] for i in range(3)]
    for m in range(0, 3):
        if m == 0:
            args.method = 'FL'
            continue
        if m == 1:
            args.method = 'CFMTL'
            args.prox = False
            args.if_clust = True
            continue
        
        if m == 2:
            args.method = 'CFMTL'
            args.prox = True
            args.if_clust = True

        if args.method == 'FL':
            loss_train = []
            pro_train = pro_final[m]
            acc_train = acc_final[m]
            w_global = Net().state_dict()

            for iter in range(args.ep):
                w_local = []
                loss_local = []
                num_clients = max(int(args.frac * args.num_clients), 1)
                clients = np.random.choice(range(args.num_clients), num_clients, replace=False)
                for id in clients:
                    mem, w, loss = Local_Update(args, w_global, dataset_train, dict_train[id], iter)
                    w_local.append(w)
                    loss_local.append(loss)
                # 多进程执行FedAvg方案
                multiprocess_wrap(func=FedAvg, world_size=2, args=(w_local,))
                w_global = torch.load("./w_avg.pth")

                loss_avg = sum(loss_local) / len(loss_local)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

                acc_test = []
                for id in range(args.num_clients):
                    acc, loss_test = Test(args, w_global, dataset_test, dict_test[id])
                    acc_test.append(acc)
                acc_avg = sum(acc_test) / len(acc_test)
                print("Testing accuracy: {:.2f}".format(acc_avg))
                acc_train.append(acc_avg.item())
                
                acc_client_num_95 = 0
                for acc_client in acc_test:
                    if acc_client >= 95:
                        acc_client_num_95 += 1
                pro_train.append(acc_client_num_95 / args.num_clients * 100)
                print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))

        if args.method == 'CFMTL':
            groups = [[i for i in range(args.num_clients)]]
            loss_train = []
            pro_train = pro_final[m]
            acc_train = acc_final[m]
            w_groups = [Net().state_dict()]

            for iter in range(args.ep):
                loss_local = []
                num_group = len(groups)

                w_local = [None for i in range(args.num_clients)]  # 所有客户端w的集合torch.save

                # 并行在客户端执行的
                for group_id in range(num_group):
                    group = groups[group_id]
                    if iter > 0:
                        num_clients = max(int(args.frac * len(group)), 1)
                        clients = np.random.choice(group, num_clients, replace=False)
                        group = clients
                    w_group = w_groups[group_id]
                    for id in group:
                        mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                        w_local[id] = w  # 添加到server端
                        loss_local.append(loss)

                if iter == 0:
                    torch.save(w_local, './w_local.pth')
                    multiprocess_wrap(Cluster_Init, world_size=2, args=(args,))
                    groups, w_groups, _, _ = torch.load("./rank0.pth")

                else:
                    torch.save(w_local, './w_local_sub.pth')
                    multiprocess_wrap(Cluster_FedAvg, world_size=2, args=(args,))
                    w_groups = torch.load("./w_groups.pth")

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

                if args.experiment == 'performance-mnist':
                    acc_client_num_95 = 0
                    for acc_client in acc_test:
                        if acc_client >= 95:
                            acc_client_num_95 += 1
                    pro_train.append(acc_client_num_95 / args.num_clients * 100)
                    print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))

        save_result(args, acc_final, pro_final, m)



def experiment_cifar(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    """在cifar数据集上的密文实现方案
    """
    acc_final = [[] for i in range(3)]
    pro_final = [[] for i in range(3)]
    for m in range(0, 3):
        if m == 0:
            args.method = 'FL'
            continue
        if m == 1:
            args.method = 'CFMTL'
            args.prox = False
            args.if_clust = True
            continue
        
        if m == 2:
            args.method = 'CFMTL'
            args.prox = True
            args.if_clust = True

        if args.method == 'FL':
            loss_train = []
            pro_train = pro_final[m]
            acc_train = acc_final[m]
            w_global = Net().state_dict()

            for iter in range(args.ep):
                w_local = []
                loss_local = []
                num_clients = max(int(args.frac * args.num_clients), 1)
                clients = np.random.choice(range(args.num_clients), num_clients, replace=False)
                for id in clients:
                    mem, w, loss = Local_Update(args, w_global, dataset_train, dict_train[id], iter)
                    w_local.append(w)
                    loss_local.append(loss)
                # 多进程执行FedAvg方案
                multiprocess_wrap(func=FedAvg, world_size=2, args=(w_local,))
                w_global = torch.load("./w_avg.pth")

                loss_avg = sum(loss_local) / len(loss_local)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

                acc_test = []
                for id in range(args.num_clients):
                    acc, loss_test = Test(args, w_global, dataset_test, dict_test[id])
                    acc_test.append(acc)
                acc_avg = sum(acc_test) / len(acc_test)
                print("Testing accuracy: {:.2f}".format(acc_avg))
                acc_train.append(acc_avg.item())
                

        if args.method == 'CFMTL':
            groups = [[i for i in range(args.num_clients)]]
            loss_train = []
            pro_train = pro_final[m]
            acc_train = acc_final[m]
            w_groups = [Net().state_dict()]

            for iter in range(args.ep):
                loss_local = []
                num_group = len(groups)

                w_local = [None for i in range(args.num_clients)]  # 所有客户端w的集合torch.save

                # 并行在客户端执行的
                for group_id in range(num_group):
                    group = groups[group_id]
                    if iter > 0:
                        num_clients = max(int(args.frac * len(group)), 1)
                        clients = np.random.choice(group, num_clients, replace=False)
                        group = clients
                    w_group = w_groups[group_id]
                    for id in group:
                        mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                        w_local[id] = w  # 添加到server端
                        loss_local.append(loss)

                if iter == 0:
                    torch.save(w_local, './w_local.pth')
                    multiprocess_wrap(Cluster_Init, world_size=2, args=(args,))
                    groups, w_groups, _, _ = torch.load("./rank0.pth")

                else:
                    torch.save(w_local, './w_local_sub.pth')
                    multiprocess_wrap(Cluster_FedAvg, world_size=2, args=(args,))
                    w_groups = torch.load("./w_groups.pth")

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

                if args.experiment == 'performance-mnist':
                    acc_client_num_95 = 0
                    for acc_client in acc_test:
                        if acc_client >= 95:
                            acc_client_num_95 += 1
                    pro_train.append(acc_client_num_95 / args.num_clients * 100)
                    print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))
        save_result(args)

def experiment_caltech101(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    """在caltech101数据集上的密文实现方案
    """
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

    

