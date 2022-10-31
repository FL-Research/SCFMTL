'''
Author: ssy
Date: 2022-10-09 05:03:25
LastEditors: ssy
LastEditTime: 2022-10-10 04:39:09
FilePath: /SCFMTL/prox_better.py
Description: 

Copyright (c) 2022 by ssy, All Rights Reserved. 
'''
from model import *
import crypten
from utils import info

def L2(_old_params, _old_w, param, args, rel):
    w = []
    for p in param:
        w.append(p.flatten())
    _w = crypten.cat(w)

    x = _w - _old_w
    x = x.norm()
    loss = x.pow(2)

    for i in range(len(_old_params)):
        _param = _old_params[i]
        x = _w - _param
        x = x.norm()
        x = x.pow(2)
        x = x.mul(args.L)
        x = x.mul(rel[i])
        loss = loss.add(x)

    return loss


def Prox(w_groups, args, rel):
    w = L2_Prox(w_groups, args, rel)
    return w


def L2_Prox(w_groups, args, rel):
    if args.dataset == 'mnist':
        Net = Crypten_Net_mnist()
    elif args.dataset == "cifar":
        Net = Crypten_Net_cifar()
    elif args.dataset == "caltech101":
        Net = crypten_net_caltech101()
    else:
        print(f"{args.dataset} undefine")
        exit(1)
    net = Net.encrypt()
    net = net.cuda()
    old_params = []
    for w in w_groups:
        tmp = []
        for k in w.keys():
            tmp.append(w[k].flatten())
        old_params.append(crypten.cat(tmp).cuda())

    w_new = []
    for i in range(len(w_groups)):
 
        w = w_groups[i]

        net.load_state_dict(w, strict=False)
        opt = crypten.optim.SGD(net.parameters(), lr=args.prox_lr, momentum=args.prox_momentum)

        _old_params = crypten.stack(old_params[:i] + old_params[i + 1:])
        rel_i = crypten.stack(rel[i])
        rel_i = rel_i.cuda()
        for iter in range(args.prox_local_ep):
            # info(_old_params.get_plain_text())
            loss = L2(_old_params, old_params[i], net.parameters(), args, rel_i)
            # if iter == 0:
            #     loss_start = copy.deepcopy(loss)
            # if iter == args.prox_local_ep - 1:
            #     loss_end = copy.deepcopy(loss)
            #     percent = (loss_end - loss_start).get_plain_text() / loss_start.get_plain_text() * 100
            #     print("Percent: {:.2f}%".format(percent))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        w_new.append(net.state_dict())

    return w_new
