# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""

from init import weight_init
from modules import *

class three_layer_cnn(object):
    def __init__(self):
        self.lr = 0.002
        self.epoch = None
        self.batch_size = 3
        self.gamma1 = 0.9
        self.beta1 = 0.1
        self.gamma2 = 0.9
        self.beta2 = 0.1
        self.gamma3 = 0.9
        self.beta3 = 0.1
        self.conv_param = {'stride': 2, 'pad': 1}
        self.bn_param = {'mode': 'train'}

    def initial(self):
        self.w1, self.b1 = weight_init(1, 1, 4, 4)
        self.w2, self.b2 = weight_init(3, 1, 4, 4)
        self.w3, self.b3 = weight_init(1, 3, 9, 9)

    def forward(self, x):
        self.var1, self.cache1 = conv_bn_relu_forward(x, self.w1, self.b1, self.gamma1, self.beta1, self.conv_param, self.bn_param)
        self.var2, self.cache2 = conv_bn_relu_forward(self.var1, self.w2, self.b2, self.gamma2, self.beta2, self.conv_param, self.bn_param)
        self.out, self.cache = conv_bn_relu_forward(self.var2, self.w3, self.b3, self.gamma3, self.beta3, self.conv_param, self.bn_param)
        return self.out

    def inference(self, x):
        self.bn_param['mode'] = 'test'
        out = self.forward(x)
        return out

    def compute_loss(self, out, y):
        pred = [0 for i in range(self.batch_size)]
        self.dout = out
        total_loss = 0
        for i in range(self.batch_size):
            pred[i] = out[i][0][0][0]
            loss = pred[i] - y[i]
            self.dout[i][0][0][0] = loss
            total_loss += abs(loss)
        ave_loss = total_loss / self.batch_size
        return ave_loss, pred

    def backward(self):
        dx3, dw3, db3, dgamma3, dbeta3 = conv_bn_relu_backward(self.dout, self.cache)
        dx2, dw2, db2, dgamma2, dbeta2 = conv_bn_relu_backward(self.var2, self.cache2)
        dx1, dw1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(self.var1, self.cache1)
        # update
        self.w1 -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.w3 -= self.lr * dw3
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2
        self.b3 -= self.lr * db3
        self.gamma1 -= self.lr * dgamma1
        self.gamma2 -= self.lr * dgamma2
        self.gamma3 -= self.lr * dgamma3
        self.beta1 -= self.lr * dbeta1
        self.beta2 -= self.lr * dbeta2
        self.beta3 -= self.lr * dbeta3