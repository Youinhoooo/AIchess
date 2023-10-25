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

if __name__ == '__main__':
    '''
    # 测试create_uci_labels()
    labels = create_uci_labels()
    print(len(labels))
    print(labels)
    '''