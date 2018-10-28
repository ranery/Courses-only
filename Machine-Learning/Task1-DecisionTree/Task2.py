# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from math import log
import csv
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from Task1 import *

def devidetraincsv():
    trainDT = csv.reader(open('TrainDT.csv', 'r'))
    dataset = []
    for line in trainDT:
        dataset.append(line)
    del(dataset[0])
    time_list = list(set([int(example[-1]) for example in dataset]))
    BSSID_list = list(set([example[0] for example in dataset]))
    label_dict = {}
    for item in dataset:
        label_dict[item[-1]] = item[2]
    pre_whole_train = []
    for i in range(0, len(time_list)):
        new_dict = {}
        for j in range(0, len(BSSID_list)):
            new_dict[BSSID_list[j]] = 0
        pre_whole_train.append(new_dict)
    for item in dataset:
        pre_whole_train[int(item[-1])-1][item[0]] = float(item[1])
    whole_train = []
    fin_label = 0
    for item in pre_whole_train:
        fin_label += 1
        new_list = []
        for keys in item.keys():
            new_list.append(item[keys])
        new_list.append(label_dict[str(fin_label)])
        whole_train.append(new_list)
    num_train = int(0.9*len(whole_train))
    num_val = len(whole_train) - num_train
    train = random.sample(whole_train, num_train)
    val = random.sample(whole_train, num_val)
    return train, val, BSSID_list

def dividetestcsv(feature):
    testDT = csv.reader(open('TestDT.csv', 'r'))
    dataset = []
    for line in testDT:
        dataset.append(line)
    del (dataset[0])
    time_list = list(set([int(example[-1]) for example in dataset]))
    BSSID_list = feature
    pre_whole_test = []
    for i in range(0, len(time_list)):
        new_dict = {}
        for j in range(0, len(BSSID_list)):
            new_dict[BSSID_list[j]] = 0
        pre_whole_test.append(new_dict)
    for item in dataset:
        pre_whole_test[int(item[-1]) - 1][item[0]] = float(item[1])
    whole_test = []
    fin_label = 0
    for item in pre_whole_test:
        fin_label += 1
        new_list = []
        for keys in item.keys():
            new_list.append(item[keys])
        whole_test.append(new_list)
    return whole_test

def discretization(dataset):
    for i in range(0, len(dataset[0])-1):
        rss_list = [example[i] for example in dataset]
        num = len(rss_list)
        for k in range(0, num):
            if rss_list[k] == 0.0:
                dataset[k][i] = 0
            else:
                dataset[k][i] = 1
    return dataset

def predict_val(dataset, feature, tree):
    label = [example[-1] for example in dataset]
    i = 0
    total_num = len(label)
    corr_num = 0
    for item in dataset:
        for keys in tree.keys():
            index = feature.index(keys)
            sub_tree = tree[keys][item[index]]
            if isinstance(sub_tree, dict):
                pred = find_leaf(item, feature, sub_tree)
            else:
                pred = sub_tree
        if pred == label[i]:
            corr_num += 1
        i += 1
    acc = corr_num / total_num
    return acc

def predict(dataset, feature, tree):
    if os.path.exists('results.csv'):
        os.remove('results.csv')
    f = open('results.csv', 'a', newline='')
    csv_write = csv.writer(f, dialect='excel')
    i = 1
    for item in dataset:
        for keys in tree.keys():
            index = feature.index(keys)
            sub_tree = tree[keys][item[index]]
            if isinstance(sub_tree, dict):
                pred = find_leaf(item, feature, sub_tree)
            else:
                pred = sub_tree
        i += 1
        result = []
        result.append(i)
        result.append(pred)
        csv_write.writerow(result)

def find_leaf(item, feature, tree):
    for keys in tree.keys():
        index = feature.index(keys)
        sub_tree = tree[keys][item[index]]
        if isinstance(sub_tree, dict):
            pred = find_leaf(item, feature, sub_tree)
        else:
            pred = sub_tree
    return pred

# def predict(dataset, tree):
#     acc_num = 0
#     total_num = len(dataset)
#     for item in dataset:
#         if item[0] in tree['BSSIDLabel']:
#             pred = tree['BSSIDLabel'][item[0]]
#             if isinstance(pred, dict):
#                 if item[1] in pred['RSSLabel']:
#                     pred = pred['RSSLabel'][item[1]]
#                     if isinstance(pred, dict):
#                         if item[2] in pred['SSIDLabel']:
#                             pred = pred['SSIDLabel'][item[2]]
#                         else:
#                             classCount = majorityResult(pred['SSIDLabel'])
#                             sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#                             pred = sortedClassCount[0][0]
#                 else:
#                     classCount = majorityResult(pred['RSSLabel'])
#                     sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#                     pred = sortedClassCount[0][0]
#         else:
#             classCount = majorityResult(tree['BSSIDLabel'])
#             sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#             pred = sortedClassCount[0][0]
#         if item[-1] == pred:
#             acc_num += 1
#     acc = acc_num / total_num
#     return acc

# def majorityResult(tree, classCount={}):
#     for keys in tree:
#         if isinstance(tree[keys], dict):
#             majorityResult(tree[keys], classCount)
#         else:
#             if tree[keys] not in classCount.keys():
#                 classCount[tree[keys]] = 0
#             classCount[tree[keys]] += 1
#     return classCount

if __name__ == '__main__':
    train_dataset, val_dataset, feature = devidetraincsv()
    test_dataset = dividetestcsv(feature)
    print('feature   ', feature)
    print('num_train ', len(train_dataset))
    print('num_val   ', len(val_dataset))
    train_dataset = discretization(train_dataset)
    myTree = createTree(train_dataset, feature.copy())
    print(myTree)
    createPlot(myTree)
    val_dataset = discretization(val_dataset)
    acc = predict_val(val_dataset, feature.copy(), myTree)
    print('accuracy  ', acc)
    test_dataset = discretization(test_dataset)
    predict(test_dataset, feature.copy(), myTree)