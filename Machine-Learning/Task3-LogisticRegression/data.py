# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import random

def parse_file(filename, has_cls=True):
    f = open(filename, 'r', encoding='gbk')
    data, label = [], []
    for line in f.readlines():
        if has_cls == True:
            data_str = line.strip().split(',')[:-1]
            data_list = []
            for data_item in data_str:
                data_list.append(float(data_item))
            data.append(data_list)
            label.append(float(line.strip().split(',')[-1]))
        else:
            data_str = line.strip().split(',')
            data_list = []
            for data_item in data_str:
                data_list.append(float(data_item))
            data.append(data_list)
    return data, label

def divide(data, label):
    num_train = int(0.8*len(data))
    train_data, train_label = [], []
    val_data, val_label = [], []
    index = random.sample(range(len(data)), num_train)
    for i in range(0, len(data)):
        if i in index:
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            val_data.append(data[i])
            val_label.append(label[i])
    return train_data, train_label, val_data, val_label

def dataset():
    data, label = parse_file('train.txt')
    train_data, train_label, val_data, val_label = divide(data, label)
    test_data, test_label = parse_file('test.txt', has_cls=False)
    print('number of train : ', len(train_data))
    print('number of val   : ', len(val_data))
    print('number of test  : ', len(test_data))
    return train_data, train_label, val_data, val_label, test_data