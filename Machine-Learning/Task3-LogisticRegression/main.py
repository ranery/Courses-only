# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from data import dataset
from logistic import logistic

# load data
train_data, train_label, val_data, val_label, test_data = dataset()
# train
logistic = logistic()
logistic.train(train_data, train_label, 30)
# val
logistic.val(val_data, val_label)
# test
logistic.test(test_data)