# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import csv
import random
import numpy as np

def getFeat():
    # train datasest
    trainDT = csv.reader(open('train.csv', 'r'))
    dataset = []
    for line in trainDT:
        dataset.append(line)
    train_feat_index = dataset[0][1:-2]
    del (dataset[0])
    # test dataset
    testDT = csv.reader(open('test.csv', 'r'))
    test_feature = []
    for line in testDT:
        test_feature.append(line)
    test_feat_index = test_feature[0][1:]
    del (test_feature[0])
    # all feature
    all_feat = list(set(train_feat_index).intersection(set(test_feat_index)))
    train_index, test_index = [], []
    for item in train_feat_index:
        if item in all_feat:
            train_index.append(all_feat.index(item))
        else:
            train_index.append(-1)
    for item in test_feat_index:
        if item in all_feat:
            test_index.append(all_feat.index(item))
        else:
            test_index.append(-1)
    num_feat = len(all_feat)
    return num_feat, train_index, test_index, dataset, test_feature

def traincsv(dataset, train_index, num_feat):
    feature = []; x = []; y = []
    for item in dataset:
        feature.append(item[1:-2])
        x.append(item[-2])
        y.append(item[-1])
    feature = [[float(j) for j in i] for i in feature]
    train_feature = []
    for i in range(len(feature)):
        _feature = list(np.zeros(num_feat))
        for j in range(len(feature[i])):
            if feature[i][j] < 0.0:
                if train_index[j] == -1:
                    pass
                else:
                    # _feature[train_index[j]] = 1.0
                    _feature[train_index[j]] = feature[i][j]
            else:
                if train_index[j] == -1:
                    pass
                else:
                    _feature[train_index[j]] = -100
        train_feature.append(_feature)
    x = [float(i) for i in x]
    y = [float(i) for i in y]
    return train_feature, x, y

def testcsv(feature, test_index, num_feat):
    for i in range(len(feature)):
        feature[i] = feature[i][1:]
    feature = [[float(j) for j in i] for i in feature]
    test_feature = []
    for i in range(len(feature)):
        _feature = list(np.zeros(num_feat))
        for j in range(len(feature[i])):
            if feature[i][j] < 0.0:
                if test_index[j] == -1:
                    pass
                else:
                    # _feature[test_index[j]] = 1.0
                    _feature[test_index[j]] = feature[i][j]
            else:
                if test_index[j] == -1:
                    pass
                else:
                    _feature[test_index[j]] = -100
        test_feature.append(_feature)
    return test_feature

def divideTrainVal(feature, x, y, ratio):
    num_dataset = len(x)
    index_train = random.sample(range(num_dataset), int(num_dataset*ratio))
    train_feature, train_x, train_y = [], [], []
    val_feature, val_x, val_y = [], [], []
    for i in range(num_dataset):
        if i in index_train:
            train_feature.append(feature[i])
            train_x.append(x[i])
            train_y.append(y[i])
        else:
            val_feature.append(feature[i])
            val_x.append(x[i])
            val_y.append(y[i])
    return train_feature, train_x, train_y, val_feature, val_x, val_y

def dataset():
    num_feat, train_index, test_index, dataset, test_feature = getFeat()
    feature, x, y = traincsv(dataset, train_index, num_feat)
    train_feature, train_x, train_y, val_feature, val_x, val_y = \
    divideTrainVal(feature[:], x[:], y[:], ratio=0.9)
    test_feature = testcsv(test_feature, test_index, num_feat)
    print('num of trainset : ', len(train_feature))
    print('num of valset   : ', len(val_feature))
    print('num of testset  : ', len(test_feature))
    print('total features  : ', num_feat)
    return train_feature, train_x, train_y, val_feature, val_x, val_y, test_feature
