# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from nb_for_spam import naive_bayes
from data import *

# dataset
train_dataset, train_cls, val_dataset, val_cls, test_dataset = dataset()
# train
nb = naive_bayes()
nb.train(train_dataset, train_cls)
nb.plot()
# val
nb.test(val_dataset, val_cls)
# test
nb.predict(test_dataset)