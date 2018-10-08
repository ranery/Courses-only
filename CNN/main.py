# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from data import *
from network import three_layer_cnn
import numpy as np

# data
train_data, test_data = loaddata()
print(train_data.keys())
print("Number of train items: %d" % len(train_data['images']))
print("Number of test items: %d" % len(test_data['labels']))
print("Edge length of picture : %f" % np.sqrt(len(train_data['images'][0])))
Class = set(train_data['labels'])
print("Total classes: ", Class)

# reshape
def imageC(data_list):
    data = np.array(data_list).reshape(len(data_list), 1, 28, 28)
    return data

# test
def test(cnn, test_batchSize):
    test_pred = []
    for i in range(int(len(test_data['images']) / test_batchSize)):
        out = cnn.inference(imageC(test_data['images'][i*test_batchSize:(i+1)*test_batchSize]))
        y = np.array(test_data['labels'][i*test_batchSize:(i+1)*test_batchSize])
        loss, pred = cnn.softmax_loss(out, y, mode='test')
        test_pred.extend(pred)
    # accuracy
    count = 0
    for i in range(len(test_pred)):
        if test_pred[i] == test_data['labels'][i]:
            count += 1
    acc = count / len(test_pred)
    return acc, loss

# train
print('Begin training ...')
cnn = three_layer_cnn()
cnn.initial()
epoch = 1
batchSize = 30
train_loss = []
train_acc = []
test_loss = []
test_acc = []
for i in range(epoch):
    for j in range(int(len(train_data['images']) / batchSize)):
    # for j in range(30):
        data = imageC(train_data['images'][j*batchSize:(j+1)*batchSize])
        label = np.array(train_data['labels'][j*batchSize:(j+1)*batchSize])
        output = cnn.forward(data)
        loss1, pred = cnn.softmax_loss(output, label)
        train_loss.append(loss1)
        if j % 100 == 0:
            # train
            count = 0
            for k in range(batchSize):
                if pred[k] == label[k]:
                    count += 1
            acc1 = count / batchSize
            train_acc.append(acc1)
        cnn.backward()
        if j % 100 == 0:
            # test
            acc2, loss2 = test(cnn, 10)
            test_loss.append(loss2)
            test_acc.append(acc2)
            print('Epoch: %d; Item: %d; Train loss: %f; Test loss: %f; Train acc: %f; Test acc: %f ' % (i, (j + 1) * batchSize, loss1, loss2, acc1, acc2))
