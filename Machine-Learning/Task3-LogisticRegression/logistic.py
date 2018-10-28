# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import numpy as np
import matplotlib.pyplot as plt
# from bigfloat import exp
import random
import csv
import os

class logistic():
    def name(self):
        return 'Logistic Model'

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(x))

    def stoGradAscent(self, data, label, numIter):
        m, n = np.shape(data)
        weight = [float(1.0) for i in range(n)]
        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                # alpha = 4 / (1.0+j+i) + 0.0001
                alpha = 0.00001
                randIndex = int(random.uniform(0, len(dataIndex)))
                h = self.sigmoid(np.sum(np.dot(data[randIndex], weight)))
                error = label[randIndex] - h
                weight += alpha * error * np.array(data[randIndex])
                del(dataIndex[randIndex])
        return weight

    def classifyVector(self, x, weight):
        pred = self.sigmoid(np.sum(np.dot(x, weight)))
        if pred > 0.5: return 1.0
        else: return 0.0

    def train(self, data, label, numIter):
        self.train_data = data
        self.train_label = label
        self.train_weight = self.stoGradAscent(data, label, numIter)

    def val(self, data, label):
        self.val_data = data
        self.val_label = label
        error = 0
        numTest = np.shape(data)[0]
        for i in range(numTest):
            pred = self.classifyVector(data[i], self.train_weight)
            # print(pred, label[i])
            if int(pred) != int(label[i]):
                error += 1
        errorRate = (float(error) / numTest)
        print('Error rate of val data : %f' % errorRate)

    def test(self, data):
        if os.path.exists('results.csv'):
            os.remove('results.csv')
        f = open('results.csv', 'a', newline='')
        csv_write = csv.writer(f, dialect='excel')
        i = 0
        for vector in data:
            result = []
            i += 1
            pred = self.classifyVector(vector, self.train_weight)
            result.append(i)
            for item in vector:
                result.append(item)
            result.append(pred)
            csv_write.writerow(result)