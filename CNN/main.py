# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from data import *
from network import three_layer_cnn

# data
train_data, test_data = loaddata()

# train
cnn = three_layer_cnn()
epoch = 200
for i in range(epoch):
	pass

# test
cls = cnn.test(test_data['images'])

# score

# figure