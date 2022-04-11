from collections import OrderedDict

import torch
import torch.distributed as dist
from prox import Prox
from cluster import *


# @run_multiprocess(world_size=2)
def Cluster_Init(w_local, args):
    rank = dist.get_rank()
    # do on client
    if w_local is None:
        w_local = torch.load('./w_local.pth')
    print(len(w_local))
    w_local_enc = encrypt_w(w_local)
    print("encrypt....")

    # do on server
    new_groups, one_hot, rel = simulation_clusters(w_local_enc, args)
    new_w_groups = cluster_avg_w(one_hot, w_local_enc)
    print("cluster over....")

    if args.prox:
        new_w_groups = Prox(new_w_groups, args, rel)

    # do on client
    for i in range(len(new_groups)):
        for j in range(len(new_groups[i])):
            new_groups[i][j] = int(new_groups[i][j].get_plain_text().item())
    new_w_groups = decrypt_w(new_w_groups)

    if rank == 0:
        torch.save((new_groups, new_w_groups, rel, one_hot), "./rank0.pth")
    else:
        torch.save((rel, one_hot), "./rank1.pth")


# @run_multiprocess(world_size=2)
def Cluster_FedAvg(w_local, one_hot_share, rel, args):
    rank = dist.get_rank()
    # encrypt
    w_local = encrypt_w(w_local)

    one_hot = one_hot_share[rank]
    rel = rel[rank]
    # do on server
    new_w_groups = cluster_avg_w(one_hot, w_local)
    if args.prox:
        new_w_groups = Prox(new_w_groups, args, rel)

    # decrypt
    new_w_groups = decrypt_w(new_w_groups)
    if rank == 0:
        torch.save(new_w_groups, "./w_groups.pth")


def FedAvg(w_local):
    w_local_en = encrypt_w(w_local)
    w_avg = [fed_avg(w_local_en)]
    w_avg = decrypt_w(w_avg)
    if dist.get_rank() == 0:
        torch.save(w_avg[0], "./w_avg.pth")


def encrypt_w(w_local):
    for i in range(len(w_local)):
        if w_local[i] is None:
            continue
        for k in w_local[i].keys():
            w_local[i][k] = crypten.cryptensor(w_local[i][k])
    return w_local


def decrypt_w(w_local_enc):
    for i in range(len(w_local_enc)):
        for k in w_local_enc[i].keys():
            w_local_enc[i][k] = w_local_enc[i][k].get_plain_text()
    return w_local_enc


def fed_avg(W, num_c=None):
    w_avg = W[0]
    if num_c is None:
        num_c = len(W)  # W中真实有的客户端数量
    for k in w_avg.keys():
        for c in range(1, len(W)):
            w_avg[k] += W[c][k]
        w_avg[k] /= num_c
    return w_avg


def cluster_avg_w(one_hot, W):
    num_g = one_hot.size()[1]
    num_c = len(W)
    mix_w_groups = [[] for _ in range(num_g)]
    num_c_per_group = [0 for _ in range(num_g)]
    for c in range(num_c):
        if W[c] is None:
            continue
        for g in range(num_g):
            mix_w = OrderedDict()  # create a new state_dict
            c_in_g_code = one_hot[c][g]
            num_c_per_group[g] += c_in_g_code
            for k in W[c].keys():
                mix_w[k] = W[c][k] * c_in_g_code
            mix_w_groups[g].append(mix_w)

    new_w_groups = []
    for g in range(num_g):
        new_w_groups.append(fed_avg(mix_w_groups[g], num_c=num_c_per_group[g]))

    return new_w_groups


# 返回客户端对应的聚合模型
def client_w(one_hot, w_groups):
    client_ws = []
    num_c = one_hot.size()[0]
    num_g = one_hot.size()[1]
    for c in range(num_c):
        mix_w = OrderedDict()
        for k in w_groups[0].keys():
            mix_w[k] = w_groups[0][k] * one_hot[c][0]
        for g in range(1, num_g):
            for k in mix_w.keys():
                mix_w[k].add_(w_groups[g][k] * one_hot[c][g])
        client_ws.append(mix_w)
    return client_ws
