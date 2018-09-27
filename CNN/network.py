# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""

from init import *
from modules import *

class three_layer_cnn(object):
    def __init__(self):
        self.lr = None
        self.epoch = None
        self.batch_size = None
        self.gamma = None
        self.beta = None
        self.conv_param = {'stride': 3, 'pad': 1}
        self.bn_param = {'mode': 'train'}

    def forward(self, x):
        w1, b1 = weight_init(f, c, hh, ww)
        self.var1, self.cache1 = conv_bn_relu_forward(x, w1, b1, self.gamma, self.beta, self.conv_param, self.bn_param)
        w2, b2 = weight_init(f, c, hh, ww)
        self.var2, self.cache2 = conv_bn_relu_forward(var1, w2, b2, self.gamma, self.beta, self.conv_param, self.bn_param)
        w3, b3 = weight_init(f, c, hh, ww)
        self.out, self.cache = conv_bn_relu_forward(var2, w3, b3, self.gamma, self.beta, self.conv_param, self.bn_param)
        return self.out

    def inference(self, x):
        self.bn_param['mode'] = 'test'
        out = self.forward(x)
        return out

    def compute_loss(self, out, y):
        pass
        return loss

    def backward(self, loss):
        dx3, dw3, db3, dgamma3, dbeta3 = conv_bn_relu_backward(self.out, self.cache)
        dx2, dw2, db2, dgamma2, dbeta2 = conv_bn_relu_backward(self.var2, self.cache2)
        dx1, dw1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(self.var1, self.cache1)
        # update
