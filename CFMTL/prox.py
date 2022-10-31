'''
Author: chenfar 825655513@qq.com
Date: 2022-10-09 05:03:25
LastEditors: ssy
LastEditTime: 2022-10-18 22:46:52
FilePath: /SCFMTL/CFMTL/prox.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
from model import *
import copy

from util import cost_time


def L2(old_params, old_w, param, args, rel, device):
    
    w = []
    for p in param:
        w.append(p)
    for i in range(len(w)):
        w[i] = torch.flatten(w[i])
    for i in range(len(old_w)):
        old_w[i] = torch.flatten(old_w[i])
    for i in range(len(old_params)):
        for j in range(len(old_params[i])):
            old_params[i][j] = torch.flatten(old_params[i][j])
    
    _w = w[0]
    for i in range(1, len(w)):
        _w = torch.cat([_w, w[i]])
    _old_w = old_w[0]
    for i in range(1, len(old_w)):
        _old_w = torch.cat([_old_w, old_w[i]])
    _old_params = []
    for i in range(len(old_params)):
        _old_param = old_params[i][0]
        for j in range(1, len(old_params[i])):
            _old_param = torch.cat([_old_param, old_params[i][j]])
        _old_params.append(_old_param)
    

    x = torch.sub(_w, _old_w)
    x = torch.norm(x, 'fro')
    x = torch.pow(x, 2)
    loss = x
    
    for i in range(len(_old_params)):
        _param = _old_params[i]
        x = torch.sub(_w, _param)
        x = torch.linalg.norm(x)
        x = torch.pow(x, 2)
        x = torch.mul(x, args.L)
        x = torch.mul(x, rel[i])
        loss = torch.add(loss, x)
        
    return loss



def Prox(w_groups, args, rel, device):
    torch.cuda.set_device(1)
    w = L2_Prox(w_groups, args, rel, device)
    return w

def L2_Prox(w_groups, args, rel, device):    
    if args.dataset == 'mnist':
        Net = Net_mnist
    if args.dataset == 'cifar':
        Net = Net_cifar
    if args.dataset == "caltech101":
        Net = net_caltech
    
    old_params = []

    net = Net().cuda()
    

    for w in w_groups:

        net.load_state_dict(w)
        
        old_params.append([])
        for p in net.parameters():
            old_params[-1].append(copy.deepcopy(p))
    
    w_n = []
    
    for i in range(len(w_groups)):
        w = w_groups[i]

        net.load_state_dict(w)       
        if args.opt == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        else:
            opt = torch.optim.SGD(net.parameters(), lr=args.prox_lr, momentum=args.prox_momentum)
        for iter in range(args.prox_local_ep):
            loss = L2(old_params[:i]+old_params[i+1:], old_params[i], net.parameters(), args, rel[i], device)
            # if iter == 0 :
            #     loss_start = copy.deepcopy(loss.item())
            # if iter == args.prox_local_ep-1:
            #     loss_end = copy.deepcopy(loss.item())
                #percent = (loss_end - loss_start) / loss_start * 100
                #print("Percent: {:.2f}%".format(percent))
            opt.zero_grad()
            loss.backward()
            opt.step()
        w_n.append(copy.deepcopy(net.state_dict()))    
    
    return w_n        
    
                
                
        

