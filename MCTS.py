# -*- coding: utf-8 -*-
"""
@author: Youinho
"""
# 蒙特卡洛树搜索

import numpy as np
import copy
from config import CONFIG

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

# 定义叶子节点
class TreeNode(object):
    def __init__(self,parent,prior_p):
        self._parent = parent
        self._children = {} #从动作到TreeNode的映射
        self._n_visits = 0  #当前节点的访问次数
        self._Q = 0 #当前节点对应动作的平均动作价值
        self._u = 0 #当前节点的置信上限（PUCT算法）
        self._p = prior_p

    def expand(self,action_priors):# 将不合法的动作概率设置为0
        for action,prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob)

    def select(self,c_puct):
        '''
        在节点中选择能提供最大Q+U的节点
        :param c_puct:
        :return:(action,next_node)的二元组
        '''
        return max(self._children.items(),key=lambda act_node:act_node[1].get_value(c_puct))

    def get_value(self,c_puct):
        '''
        计算并返回此节点的值，它是节点评估Q和此节点的先验的组合
        c_puct:控制相对影响(0,inf)
        :param c_puct:
        :return:
        '''
        self._u = (c_puct * self._p * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self,leaf_value):
        '''
        从叶节点评估中更新节点值
        :param leaf_value:这个子节点的评估值来自当前玩家的视角
        :return:
        '''
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 使用递归的方法对所有节点（当前节点对应的支线）进行一次更新
    def update_recursive(self,leaf_value):
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        # 检查是否是叶节点，即没有被拓展的节点
        return self._children == {}

    def is_root(self):
        return self._parent is None






