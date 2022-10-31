"""实现密文版的各个方案
"""
import os

import torch

from CFMTL.data import *
from CFMTL.local import Local_Update, Local_Update2
from CFMTL.test import Test
from CFMTL.model import *
import argparse
from aggre import *
import numpy as np
from torchvision import datasets, transforms
from crypten.mpc import multiprocess_wrap

from experiment import experiment_caltech101, experiment_cifar, experiment_mnist


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--iid', type=str, default='iid')  # non-iid
parser.add_argument('--ratio', type=float, default=0.25)

parser.add_argument('--method', type=str, default='CFMTL')
parser.add_argument('--ep', type=int, default=2)
parser.add_argument('--local_ep', type=int, default=2)
parser.add_argument('--frac', type=float, default=0.2)
parser.add_argument('--num_batch', type=int, default=64)

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.5)

parser.add_argument('--num_clients', type=int, default=200)
parser.add_argument('--clust', type=int, default=20)
parser.add_argument('--if_clust', type=bool, default=True)

parser.add_argument('--prox', type=bool, default=True)
parser.add_argument('--R', type=str, default='L2')
parser.add_argument('--prox_local_ep', type=int, default=5)
parser.add_argument('--prox_lr', type=float, default=0.01)
parser.add_argument('--prox_momentum', type=float, default=0.5)
parser.add_argument('--L', type=float, default=1)
parser.add_argument('--dist', type=str, default='L2')

parser.add_argument('--experiment', type=str, default='performance-mnist')

import multiprocessing as mp





# Clustered Secure Sparse Aggregation for Federated Learning on Non-IID Data
# Secure Sparse Aggregation with hierarchical clustering for Federated Learning on Non-IID Data
if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid == 'iid':
            dict_train, dict_test = mnist_iid(dataset_train, dataset_test, args.num_clients, 10)
        elif args.iid == 'non-iid':
            dict_train, dict_test = mnist_non_iid(dataset_train, dataset_test, args.num_clients, 10, args.ratio)
        else:  # args.iid == 'non-iid-single_class'
            dict_train, dict_test = mnist_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 10)
        Net = Net_mnist
    
    elif args.dataset == 'cifar':  # args.dataset == 'cifar'
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar/', train=False, download=True, transform=trans_cifar)
        if args.iid == 'iid':
            dict_train, dict_test = cifar_iid(dataset_train, dataset_test, args.num_clients, 10)
        elif args.iid == 'non-iid':
            dict_train, dict_test = cifar_non_iid(dataset_train, dataset_test, args.num_clients, 10, args.ratio)
        else:  # args.iid == 'non-iid-single_class':
            dict_train, dict_test = cifar_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 10)
        Net = Net_cifar

    elif args.dataset == "caltech101":
        from CFMTL.preprocess_data import get_dataset_caltech101
        from CFMTL.model import net_caltech
        from model import crypten_net_caltech101

        dataset_train, dataset_test = get_dataset_caltech101(data_dir="data/caltech-101/101_ObjectCategories")

        if args.iid == 'iid':
            dict_train, dict_test = divided_data_iid(dataset_train, dataset_test, args.num_clients, num_class=101)
        elif args.iid == 'non-iid':
            dict_train, dict_test = divided_data_non_iid(dataset_train, dataset_test, args.num_clients, 101, args.ratio)
        elif args.iid == 'non-iid-single_class':
            dict_train, dict_test = divided_data_non_iid_single_class(dataset_train, dataset_test, args.num_clients, 101)
        
        Net = net_caltech()
        # Net = crypten_net_caltech101()

    if args.experiment == 'performance-mnist' :

        result = experiment_mnist(
            args, 
            dataset_train, 
            dataset_test, 
            dict_train, 
            dict_test, 
            Net, 
            device
        )

    elif args.experiment == 'performance-cifar':
        result = experiment_cifar(
            args, 
            dataset_train, 
            dataset_test, 
            dict_train, 
            dict_test, 
            Net, 
            device
        )

    elif args.experiment == "performace-caltech101":
        print("="*25 + "> Start experiment on caltech101")
        # args.method = 'CFMTL'
        # args.prox = False
        # args.if_clust = True
        acc_train = experiment_caltech101(
            args, 
            dataset_train, 
            dataset_test, 
            dict_train, 
            dict_test, 
            Net, 
            device
        )
        print("save result")
        if args.iid == "iid":
            filename = f'log_secure/result/{args.experiment}-iid-secure.npy'
        elif args.iid == "non-iid":
            filename = f'log_secure/result/{args.experiment}-noniid-{args.ratio}-secure.npy'
        else:  # non-iid-single_class
            filename = f'log_secure/result/{args.experiment}-non-iid-single_class-secure.npy'
        np.save(filename, acc_train)
        print("="*25 + ">Finish experiment on caltech101")
        
# python
