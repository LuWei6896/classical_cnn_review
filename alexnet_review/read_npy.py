#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact: wendell@sina.com
@software: wendell
@file: .py
@time: 18-11-19 下午3:19
@desc:
'''
import numpy as np

def test_load(path):
    data_dict = np.load(path,encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        name = key
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('name:',name)
        print('weights:',weights.shape)
        print('biases:',biases.shape)



if __name__=='__main__':
    test_load('bvlc_alexnet.npy')