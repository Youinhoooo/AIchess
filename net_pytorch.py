# -*- coding: utf-8 -*-
"""
@author: Youinho
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# 设置学习率
def set_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 构建残差层
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        # 标准化
        self.conv1_bn = nn.BatchNorm2d(256,)
        # 激活函数
        self.conv1_act = nn.ReLU()
        # 卷积层
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
        # 标准化
        self.conv2_bn = nn.BatchNorm2d(256,)
        # 激活函数
        self.conv2_act = nn.ReLU()
    def forward(self,x):
        y = self.conv1(x)
        y = self.conv1_bn(x)
        y = self.conv1_act(x)
        y = self.conv2(x)
        y = self.conv1_bn(x)
        y = x + y
        return self.conv2_act(y)

# 网络搭建
class Net(nn.Module):
    def __init__(self,num_channels=256,num_res_blocks=7):
        super(Net,self).__init__()

        # 卷积块
        self.conv_block = nn.Conv2d(in_channels=9,out_channels=num_channels,kernel_size=(3,3))
        self.conv_block_bn = nn.BatchNorm2d(256)
        self.conv_block_act = nn.ReLU()
        # 残差块(19或39，先试试7)
        self.res_blocks = nn.ModuleList([ResBlock() for _ in range(num_res_blocks)])
        # 两个单独的“头”
        # 策略头
        self.policy_conv = nn.Conv2d(in_channels=num_channels,out_channels=16,kernel_size=(1,1),stride=(1,1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16*9*10,2086)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels,out_channels=8,kernel_size=(1,1),stride=(1,1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8*9*10,256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256,1)



