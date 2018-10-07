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
cnn = three_layer_cnn()
epoch = 20
batchSize = 3
for i in range(epoch):
    for j in range(int(len(train_data['images']) / batchSize)):
        data = imageC(train_data['images'][j*batchSize:(j+1)*batchSize])
        label = np.array(train_data['labels'][j*batchSize:(j+1)*batchSize])
        cnn.initial()
        output = cnn.forward(data)
        loss, pred = cnn.compute_loss(output, label)
        print('Epoch: %d; Item: %d; Loss: %f' % (i, (j+1)*batchSize, loss))
        print(pred, label)
        cnn.backward()

# test
# cls = cnn.test(test_data['images'])

# score

# figure