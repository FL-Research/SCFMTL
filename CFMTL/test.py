'''
Author: chenfar 825655513@qq.com
Date: 2022-10-09 05:03:25
LastEditors: chenfar 825655513@qq.com
LastEditTime: 2022-10-17 07:55:08
FilePath: /SCFMTL/CFMTL/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

# from CFMTL.model import *
from model import Net_mnist, Net_cifar, net_caltech
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from util import cost_time
import torch

class DatasetFolder(Dataset):
    def __init__(self, dataset, ids):
        self.dataset = dataset
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        images, labels = self.dataset[self.ids[item]]
        return images, labels

# @cost_time
def Test(args, w, dataset=None, ids=None):
    dataloader = DataLoader(DatasetFolder(dataset, ids), batch_size=args.num_batch, shuffle=True, num_workers=4)

    if args.dataset == 'mnist':
        Net = Net_mnist
    if args.dataset == 'cifar':
        Net = Net_cifar
    if args.dataset == "caltech101":
        Net = net_caltech
        

    net = Net().cuda()
    net.load_state_dict(w)
    
    net.eval()

    loss_test = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
     
            images, labels = images.cuda(), labels.cuda()
            log_probs = net(images)
            loss_test += F.cross_entropy(log_probs, labels, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

        loss_test /= len(dataloader.dataset)
        acc = 100.00 * correct / len(dataloader.dataset)

    return acc, loss_test
