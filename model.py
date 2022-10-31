'''
Author: ssy
Date: 2022-10-09 05:03:25
LastEditors: ssy
LastEditTime: 2022-10-19 23:06:01
FilePath: /SCFMTL/model.py
Description: 

Copyright (c) 2022 by ssy, All Rights Reserved. 
'''

import crypten.nn as nn
import torch
import pretrainedmodels
import torch.nn.functional as F

class Crypten_Net_mnist(nn.Module):
    def __init__(self):
        super(Crypten_Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.max_pool2d(self.conv1(x), 2).relu()
        x = self.max_pool2d(self.conv2_drop(self.conv2(x)), 2).relu()
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x).relu()
        x = self.dropout(x, training=self.training)
        x = self.fc2(x)
        return x.log_softmax(dim=1)


class Crypten_Net_cifar(nn.Module):
    def __init__(self):
        super(Crypten_Net_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
  

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x.log_softmax(dim=1)


class Crypten_Net_caltech101(nn.Module):
    def __init__(self):
        super(Crypten_Net_caltech101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(5, 5)
        
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 512, 3)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 101)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = self.pool(self.conv3(x).relu())
        x = x.view(x.size(0), -1)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x.log_softmax(dim=1)




class ResNet34(torch.nn.Module):
    """resnet34 模型定义
    """
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        
        # change the classification layer
        # len(lb.classes_) = 101
        self.l0 = torch.nn.Linear(512, 101)
        # self.dropout = torch.nn.Dropout2d(0.4)
        self.dropout = torch.nn.Dropout(0.4)
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0



def net_caltech():
    """获取caltech数据集所对应的模型
    """
    return ResNet34(pretrained=True)

def crypten_net_caltech101():
    net = net_caltech()
    dumpy_input = torch.ones(size=[16, 3, 224, 224])
    crypten_net = nn.from_pytorch(net, dumpy_input)

    return crypten_net


def test_crypten_model_load():
    net = crypten_net_caltech101()
    net = net.encrypt()
    net = net.cuda()


