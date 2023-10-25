# -*- coding: utf-8 -*-
"""
@author: Youinho
"""

import numpy as np

y_axis = '9876543210'
x_axis = 'abcdefghi'

#创建所有合法走子uci,size 2086
def create_uci_labels():
    labels = []
    column = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    corrdinates = [] #存储所有位置坐标

    # 士的位移
    Advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    # 将帅的位移
    Bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']
    # 棋盘上的所有坐标
    for c1 in range(9):
        for r1 in range(10):
            arrive = (c1,r1)
            corrdinates.append(arrive)

    # 马走日
    for corrdinate in corrdinates:
        temp_move_corrdinates = [(corrdinate[0]+a,corrdinate[1]+b) for (a,b) in
                                     [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
        for item in temp_move_corrdinates:
            if item[0] in range(9) and item[1] in range(10):
                move = column[corrdinate[0]] + row[corrdinate[1]] + column[item[0]] + row[item[1]]
                labels.append(move)

    # 车、炮、兵
    for corrdinate_1 in corrdinates:
        row_corrdinates = [(t,corrdinate_1[1]) for t in range(9)]
        column_corrdinates = [(corrdinate_1[0],t) for t in range(10)]
        row_column_corrdinates = row_corrdinates + column_corrdinates
        for item in row_column_corrdinates:
            if corrdinate_1 !=item:
                move = column[corrdinate_1[0]] + row[corrdinate_1[1]] + column[item[0]] + row[item[1]]
                labels.append(move)

    for label in Advisor_labels:
        labels.append(label)
    for label in Bishop_labels:
        labels.append(label)

    return labels

'''
y_axis = '9876543210'
x_axis = 'abcdefghi'
'''

# 定义棋子类
class piece(object):
    def __init__(self):
        # 红色棋子
        self.r_ju = np.array([1, 0, 0, 0, 0, 0, 0])
        self.r_ma = np.array([0, 1, 0, 0, 0, 0, 0])
        self.r_xiang = np.array([0, 0, 1, 0, 0, 0, 0])
        self.r_shi = np.array([0, 0, 0, 1, 0, 0, 0])
        self.r_jiang = np.array([0, 0, 0, 0, 1, 0, 0])
        self.r_pao = np.array([0, 0, 0, 0, 0, 1, 0])
        self.r_bing = np.array([0, 0, 0, 0, 0, 0, 1])
        # 黑色棋子
        self.b_ju = np.array([-1, 0, 0, 0, 0, 0, 0])
        self.b_ma = np.array([0, -1, 0, 0, 0, 0, 0])
        self.b_xiang = np.array([0, 0, -1, 0, 0, 0, 0])
        self.b_shi = np.array([0, 0, 0, -1, 0, 0, 0])
        self.b_jiang = np.array([0, 0, 0, 0, -1, 0, 0])
        self.b_pao = np.array([0, 0, 0, 0, 0, -1, 0])
        self.b_bing = np.array([0, 0, 0, 0, 0, 0, -1])
        # 创建字典
        self.create_dict()
    #建立棋子名字到序列的字典
    def create_dict(self):
        self.dict = {'红车':self.r_ju, '红马':self.r_ma, '红象':self.r_xiang, '红士':self.r_shi,
                     '红将':self.r_jiang, '红炮':self.r_pao, '红兵':self.r_bing,
                     '黑车': self.b_ju, '黑马': self.b_ma, '黑象': self.b_xiang, '黑士': self.b_shi,
                     '黑将': self.b_jiang, '黑炮': self.b_pao, '黑兵': self.b_bing,
                     }
    # 棋子序列向棋子名的映射
    def array2name(self,array):
        for key,value in self.dict.items():
            if (array == value).all():
                return key
        return 0
    # 棋子名向棋子序列的映射
    def name2array(self,name):
        return self.dict[name]

# 定义棋盘
'''
9车马象士将士象马车
8
7 炮         炮
6兵  兵  兵  兵  兵
5
4
3兵  兵  兵  兵  兵
2 炮         炮
1 
0车马象士将士象马车
 abcdefghi
 012345678
'''
class Board(object):
    def __init__(self):
        self.y_axis = '9876543210'
        self.x_axis = 'abcdefghi'
        self.height = 10
        self.width = 9
        self.board = np.zeros((self.height, self.width, 7))
        self.p = piece()
        self.init_board()

    def init_board(self):
        self.board[0, 0], self.board[0, 8] = self.p.r_ju, self.p.r_ju
        self.board[0, 1], self.board[0, 7] = self.p.r_ma, self.p.r_ma
        self.board[0, 2], self.board[0, 6] = self.p.r_xiang, self.p.r_xiang
        self.board[0, 3], self.board[0, 5] = self.p.r_shi, self.p.r_shi
        self.board[2, 1], self.board[2, 7] = self.p.r_pao, self.p.r_pao
        self.board[3, 0], self.board[3, 2], self.board[3, 4], self.board[3, 6], self.board[
            3, 8] = self.p.r_bing, self.p.r_bing, self.p.r_bing, self.p.r_bing, self.p.r_bing
        self.board[0, 4] = self.p.r_jiang
        self.board[9, 0], self.board[9, 8] = self.p.b_ju, self.p.b_ju
        self.board[9, 1], self.board[9, 7] = self.p.b_ma, self.p.b_ma
        self.board[9, 2], self.board[9, 6] = self.p.b_xiang, self.p.b_xiang
        self.board[9, 3], self.board[9, 5] = self.p.b_shi, self.p.b_shi
        self.board[7, 1], self.board[7, 7] = self.p.b_pao, self.p.b_pao
        self.board[6, 0], self.board[6, 2], self.board[6, 4], self.board[6, 6], self.board[
            6, 8] = self.p.b_bing, self.p.b_bing, self.p.b_bing, self.p.b_bing, self.p.b_bing
        self.board[9, 4] = self.p.b_jiang

    # 展示棋盘和棋子
    def display(self):
        piece_list = []
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                piece_name = self.p.array2name(self.board[i][j])
                piece_list.append(piece_name)
        temp_list = []
        all_piece_list = []
        i = 0
        for piece in piece_list:
            if piece == 0:
                piece = '一一'
            temp_list.append(piece)
            #print(temp_list)
            i = i+1
            if i%9 == 0:
                all_piece_list.insert(0,temp_list)
                temp_list = []
        for piece_list in all_piece_list:
            print(piece_list)

# 将棋子的一步位移转换为字符串
def get_move(board1,board2):
    column = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    row = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    height = 10
    width = 9
    for i in range(height):
        for j in range(width):
            if (board1[i,j] == board2[i,j]).all() != 1:
                print((i,j))
                if (board1[i,j]==0).all() != 1:
                    first = column[j] + row[i]
                else:
                    second = column[j] + row[i]

    return first+second

if __name__ == '__main__':
    '''
    # 测试create_uci_labels()
    labels = create_uci_labels()
    print(len(labels))
    print(labels)
    # 测试piece类
    p = piece()
    a = p.array2name(np.array([0, 0, 0, 0, 0, 0, -1]))
    print(a)
    b = p.name2array('红车')
    print(b)
    # 测试Board类
    b = Board()
    print(b.display())
    # 测试get_move()
    b1 = Board()
    b2 = Board()
    board1 = b1.board
    board2 = b2.board
    board2[6, 1] = np.array([0, 0, 0, 0, 0, 1, 0])
    board2[2, 1] = np.array([0, 0, 0, 0, 0, 0, 0])
    b2.board[6, 1] = np.array([0, 0, 0, 0, 0, 1, 0])
    b2.board[2, 1] = np.array([0, 0, 0, 0, 0, 0, 0])
    b2.display()
    print(get_move(board1,board2))
    '''
