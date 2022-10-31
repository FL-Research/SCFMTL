'''
Author: ssy
Date: 2022-10-13 03:06:30
LastEditors: chenfar 825655513@qq.com
LastEditTime: 2022-10-18 09:54:41
FilePath: /SCFMTL/CFMTL/experiment.py
Description: 根据不同的数据集对所提出的方案进行实验

Copyright (c) 2022 by ssy, All Rights Reserved. 
'''

from tokenize import cookie_re
import numpy as np
import copy

# 自定义方法和类
from cluster import Cluster
from fedavg import FedAvg
from local import Local_Update, Local_Update3
from prox import Prox
from test import Test
from tqdm import tqdm



def experiment_mnist(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    """对于mnist数据集实验的主要实现
    """
    # 记录全局模型进度
    acc_final = [[] for i in range(3)]
    # 记录客户端训练精度达到95%所占的比例
    pro_final = [[] for i in range(3)]  
    for m in range(1, 3):
        if m == 0:
            args.method = 'FL'
        if m == 1:
            args.method = 'CFMTL'
            args.prox = False
            args.if_clust = True
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
                w_global = FedAvg(w_local)
                loss_avg = sum(loss_local) / len(loss_local)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

                acc_test = []
                for id in range(args.num_clients):
                    acc, loss_test = Test(args, w_global, dataset_test, dict_test[id])
                    acc_test.append(acc)
                acc_avg = sum(acc_test) / len(acc_test)
                print("Testing accuracy: {:.2f}".format(acc_avg))
                acc_train.append(acc_avg)
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
            w_global = Net().state_dict()
            w_groups = [w_global]
            for iter in range(args.ep):
                loss_local = []
                num_group = len(groups)
                for group_id in range(num_group):
                    if iter == 0:
                        group = groups[group_id]
                    else:
                        group = groups[group_id]
                        num_clients = max(int(args.frac * len(group)), 1)
                        clients = np.random.choice(group, num_clients, replace=False)
                        group = clients
                    w_group = w_groups[group_id]
                    w_local = []
                    for id in group:
                        mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                        w_local.append(w)
                        loss_local.append(loss)
                    if iter == 0:
                        groups, w_groups, rel = Cluster(group, w_local, args)
                    else:
                        w_groups[group_id] = FedAvg(w_local)

                if len(groups) > 1 and args.prox is True:
                    w_groups = Prox(w_groups, args, rel)

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
                acc_train.append(acc_avg)
                acc_client_num_95 = 0
                for acc_client in acc_test:
                    if acc_client >= 95:
                        acc_client_num_95 += 1
                pro_train.append(acc_client_num_95 / args.num_clients * 100)
                print("Testing proportion: {:.1f}".format(acc_client_num_95 / args.num_clients * 100))
    record = []
    record.append(copy.deepcopy(acc_final))
    record.append(copy.deepcopy(pro_final))
    return record


def experiment_cifar(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    """对于cifar数据集实验的主要实现
    """
    acc_final = [[] for i in range(3)]
    for m in range(3):
        """对三种方案依次执行"""
        if m == 0:
            args.method = 'FL'      # 表示方案Fedavg
        if m == 1:
            args.method = 'CFMTL'   # 表示方案CFL
            args.prox = False       # 不进行组间学习
            args.if_clust = True
        if m == 2:
            args.method = 'CFMTL'   # 表示方案CFTL
            args.prox = True        # 进行组间学习
            args.if_clust = True

        if args.method == 'FL':
            loss_train = []
            acc_train = acc_final[m]
            w_global = Net().state_dict()
            for iter in tqdm(range(args.ep)):
                w_local = []
                loss_local = []
                num_clients = max(int(args.frac * args.num_clients), 1)
                clients = np.random.choice(range(args.num_clients), num_clients, replace=False)
                for id in clients:
                    mem, w, loss = Local_Update(args, w_global, dataset_train, dict_train[id], iter)
                    w_local.append(w)
                    loss_local.append(loss)
                w_global = FedAvg(w_local)
                loss_avg = sum(loss_local) / len(loss_local)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

                acc_test = []
                for id in range(args.num_clients):
                    acc, loss_test = Test(args, w_global, dataset_test, dict_test[id])
                    acc_test.append(acc)
                acc_avg = sum(acc_test) / len(acc_test)
                print("Testing accuracy: {:.2f}".format(acc_avg))
                acc_train.append(acc_avg)

        if args.method == 'CFMTL':
            groups = [[i for i in range(args.num_clients)]]
            loss_train = []
            acc_train = acc_final[m]
            w_global = Net().state_dict()
            w_groups = [Net().state_dict()]
            for iter in range(args.ep):
                loss_local = []
                num_group = len(groups)
                for group_id in range(num_group):
                    if iter == 0:
                        group = groups[group_id]
                    else:
                        group = groups[group_id]
                        num_clients = max(int(args.frac * len(group)), 1)
                        clients = np.random.choice(group, num_clients, replace=False)
                        group = clients
                    w_group = w_groups[group_id]
                    w_local = []
                    for id in group:
                        mem, w, loss = Local_Update(args, w_group, dataset_train, dict_train[id], iter)
                        w_local.append(w)
                        loss_local.append(loss)
                    if iter == 0:
                        groups, w_groups, rel = Cluster(group, w_local, args)
                    else:
                        w_groups[group_id] = FedAvg(w_local)

                if len(groups) > 1 and args.prox is True:
                    w_groups = Prox(w_groups, args, rel)

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
                acc_train.append(acc_avg)
    record = []
    record.append(copy.deepcopy(acc_final))

def experiment_caltech101(args, dataset_train, dataset_test, dict_train, dict_test, Net, device):
    """对caltech101数据集进行实验
    """
    acc_final = [[] for i in range(3)]
    for m in range(3):
        """对三种方案依次执行"""
        if m == 0:
            print("="*20 + ">Start FedAvg scheme")
            args.method = 'FL'      # 表示方案Fedavg
            continue
            
        if m == 1:
            print("="*20 + ">Start CFL scheme")
            args.method = 'CFMTL'   # 表示方案CFL
            args.prox = False       # 不进行组间学习
            args.if_clust = True
     
            
        if m == 2:
            print("="*20 + ">Start CFMTL scheme")
            args.method = 'CFMTL'   # 表示方案CFTL
            args.prox = True        # 进行组间学习
            args.if_clust = True
            # continue
            
        if args.method == 'FL':
            """对fedavg方案进行实验
            """
            # print("Start FedAvg")
            loss_train = []
            acc_train = acc_final[m]
            w_global = Net.state_dict()
            for iter in tqdm(range(args.ep)):
                w_local = []
                loss_local = []
                num_clients = max(int(args.frac * args.num_clients), 1)
                clients = np.random.choice(range(args.num_clients), num_clients, replace=False)
                for id in clients:
                    mem, w, loss = Local_Update3(args, w_global, dataset_train, dict_train[id], iter)
                    w_local.append(w)
                    loss_local.append(loss)
                w_global = FedAvg(w_local)
                loss_avg = sum(loss_local) / len(loss_local)
                print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
                loss_train.append(loss_avg)

                acc_test = []
                # import pdb
                # pdb.set_trace()
                for id in range(args.num_clients):
                    acc, loss_test = Test(args, w_global, dataset_test, dict_test[id])
                    acc_test.append(acc)
                acc_avg = sum(acc_test) / len(acc_test)
                print("Testing accuracy: {:.2f}".format(acc_avg))
                acc_train.append(acc_avg)

        if args.method == 'CFMTL':
            
            groups = [[i for i in range(args.num_clients)]]
            loss_train = []
            acc_train = acc_final[m]
            w_global = Net.state_dict()
            w_groups = [Net.state_dict()]
            for iter in tqdm(range(args.ep)):
                loss_local = []
                num_group = len(groups)
                for group_id in range(num_group):
                    if iter == 0:
                        group = groups[group_id]
                    else:
                        group = groups[group_id]
                        num_clients = max(int(args.frac * len(group)), 1)
                        clients = np.random.choice(group, num_clients, replace=False)
                        group = clients
                    w_group = w_groups[group_id]
                    w_local = []
                    for id in group:
                        mem, w, loss = Local_Update3(args, w_group, dataset_train, dict_train[id], iter)
                        w_local.append(w)
                        loss_local.append(loss)
                        
                    if iter == 0:
                        groups, w_groups, rel = Cluster(group, w_local, args)
                    else:
                        w_groups[group_id] = FedAvg(w_local)

                if len(groups) > 1 and args.prox is True:
                    w_groups = Prox(w_groups, args, rel, device)

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
                acc_train.append(acc_avg)
    record = []
    record.append(copy.deepcopy(acc_final))

    return record
