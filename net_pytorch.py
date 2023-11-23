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
from config import CONFIG
from torch.cuda.amp import autocast

# 设置学习率
def set_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 构建残差层
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),stride=(1,1),padding=1)
        # 标准化
        self.conv1_bn = nn.BatchNorm2d(256,)
        # 激活函数
        self.conv1_act = nn.ReLU()
        # 卷积层
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),stride=(1,1),padding=1)
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
        self.conv_block = nn.Conv2d(in_channels=9,out_channels=num_channels,kernel_size=(3,3),stride=(1,1),padding=1)
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

    def forward(self,x):
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)

        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy,[-1,16*10*9])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)

        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value,[-1,8*10*9])
        value = self.value_fc1(value)
        value = self.value_act2(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy,value

# 策略值网络，用来进行模型的训练
class PolicyValueNet:
    def __init__(self,model_file=None,use_gpu=True,device='cuda'):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3    #L2正则化
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,
                                          weight_decay=self.l2_const)
        # 加载模型
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self,state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs,value = self.policy_value_net(state_batch)
        log_act_probs,value = log_act_probs.cpu(),value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs,value.detach().numpy()

    # 输入棋盘，返回每一个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self,board):
        self.policy_value_net.eval()
        # 获取合法动作列表
        legal_positions = board.availables  #available还没有定义！
        current_state = np.ascontiguousarray(board.current_state().reshape(-1,9,10,9)).astype('float16')    #current_state还没有定义！
        current_state = torch.as_tensor(current_state).to(self.device)
        # 使用神经网络进行预测
        with autocast():#半精度fp16
            log_act_probs,value = self.policy_value_net(current_state)
        log_act_probs,value = log_act_probs.cpu(),value.cpu()
        act_probs = np.exp(log_act_probs.numpy().flatten()) if CONFIG['use_frame'] == 'paddle' else \
            np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        # 只取出合法动作
        act_probs = zip(legal_positions,act_probs[legal_positions])
        # 返回动作概率以及状态价值
        return act_probs,value.detach().numpy()

    # 保存模型
    def save_model(self,model_file):
        torch.save(self.policy_value_net.state_dict(),model_file)

    # 执行一步训练
    def train_step(self,state_batch,mcts_probs,winner_batch,lr=0.002):
        self.policy_value_net.train()
        # 包装变量
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)
        # 清零梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        # forward
        log_act_probs,value = self.policy_value_net(state_batch)
        value = torch.reshape(value,shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value,target=winner_batch)
        # 策略损失
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs,dim=1))    # 希望两个方向向量越一致越好
        # 总的损失,l2惩罚已经包含在优化器内
        loss = value_loss + policy_loss
        # backward和优化
        loss.backward()
        self.optimizer().step()
        # 计算策略的熵，用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs)*log_act_probs,dim=1)
            )

        return loss.detach().cpu().numpy(),entropy.detach().cpu().numpy()

if __name__ == '__main__':
    net = Net().to('cuda')
    test_data = torch.ones([8,9,10,9]).to('cuda')
    x_act,x_val = net(test_data)
    print(x_act.shape)
    print(x_val.shape)


