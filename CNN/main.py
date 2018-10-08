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

# train
print('Begin training ...')
cnn = three_layer_cnn()
cnn.initial()
epoch = 1
batchSize = 30
for i in range(epoch):
    for j in range(int(len(train_data['images']) / batchSize)):
    # for j in range(30):
        data = imageC(train_data['images'][j*batchSize:(j+1)*batchSize])
        label = np.array(train_data['labels'][j*batchSize:(j+1)*batchSize])
        output = cnn.forward(data)
        loss, pred = cnn.compute_loss(output, label)
        if j % 100 == 0:
            count = 0
            for k in range(batchSize):
                if pred[k] == label[k]:
                    count += 1
            train_acc = count / batchSize
            print('Epoch: %d; Item: %d; Loss: %f; Train acc: %f ' % (i, (j + 1) * batchSize, loss, train_acc))
        cnn.backward()

# test
print('Begin testing ...')
batchSize = 10
test_pred = []
for i in range(int(len(test_data['images']) / batchSize)):
    output = cnn.inference(imageC(test_data['images'][i*batchSize:(i+1)*batchSize]))
    label = np.array(test_data['labels'][i*batchSize:(i+1)*batchSize])
    loss, pred = cnn.compute_loss(output, label)
    test_pred.extend(pred)

# accuracy
count = 0
for i in range(len(test_pred)):
    if test_pred[i] == test_data['labels'][i]:
        count += 1
acc = count / len(test_pred)
print('Accuracy for 3-layers convolutional neural networks: %f' % acc)

# figure