# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""

from init import weight_init
from modules import *

class three_layer_cnn(object):
    def __init__(self):
        self.epoch = None
        self.config = {'learning_rate': 1e-4,'decay_rate': 0.99, 'epsilon': 1e-8}
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
        self.w2, self.b2 = weight_init(2, 1, 4, 4)
        self.w3, self.b3 = weight_init(10, 2, 9, 9)

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
        # pred
        pred = []
        for i in range(len(out)):
            score = out[i].reshape(10, 1)
            predict = np.where(score == max(score))[0]
            if len(predict) > 1:
                pred.append(int(predict[0]))
            else:
                pred.append(int(predict))

        # svm loss
        correct_class_score = out[range(len(out)), y].reshape(len(out), 1)
        margin = np.maximum(0.0, out - correct_class_score + 1)
        margin[range(len(out)), y] = 0.0
        loss = np.sum(margin) / len(out)
        # loss += 0.5 * (np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2) + np.sum(self.w3 * self.w3))

        # grad
        margin[margin > 0] = 1
        margin[range(len(out)), y] = -1 * np.sum(margin, axis=1)
        self.dout = margin
        return loss, pred

    def backward(self):
        dx3, dw3, db3, dgamma3, dbeta3 = conv_bn_relu_backward(self.dout, self.cache)
        dx2, dw2, db2, dgamma2, dbeta2 = conv_bn_relu_backward(dx3, self.cache2)
        dx1, dw1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(dx2, self.cache1)
        # update
        self.w1 = self.rmsprop(self.w1, dw1)
        self.w2 = self.rmsprop(self.w2, dw2)
        self.w3 = self.rmsprop(self.w3, dw3)
        self.b1 = self.rmsprop(self.b1, db1)
        self.b2 = self.rmsprop(self.b2, db2)
        self.b3 = self.rmsprop(self.b3, db3)
        self.gamma1 = self.rmsprop(self.gamma1, np.sum(dgamma1))
        self.gamma2 = self.rmsprop(self.gamma2, np.sum(dgamma2))
        self.gamma3 = self.rmsprop(self.gamma3, np.sum(dgamma3))
        self.beta1 = self.rmsprop(self.beta1, np.sum(dbeta1))
        self.beta2 = self.rmsprop(self.beta2, np.sum(dbeta2))
        self.beta3 = self.rmsprop(self.beta3, np.sum(dbeta3))

    def rmsprop(self, x, dx, config=None):
        if config is None: config = self.config
        config['cache'] = np.zeros_like(x)

        next_x = None
        config['cache'] += dx ** 2
        next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + 1e-7)

        return next_x