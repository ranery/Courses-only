# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
from data import *
from SVM import SVM

file_path = os.getcwd()

def Task1():
    # load and visulize data
    svm = SVM()
    print('Loading and Visualizing Data ...')
    X, y = loadData(file_path + '\data1_Task.mat')
    plotData(X, y, title='Task 1')

    print('Program paused. Press enter to continue.')
    input()

    # Training
    print('Training SVM with RBF kernel ...')
    param = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    acc = []
    for C in param:
        for sigma in param:
            model = svm.svmTrain_SMO(X, y, C, kernal_function='gaussian', K_matrix=svm.gaussianKernel(X, sigma))
            pred = svm.svmPredict(model, np.mat(X), sigma)
            acc.append(1 - np.sum(abs(pred - y)) / len(y))
    print(acc)
    f = open('results.txt', 'a+')
    f.write('sigma\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\nC\n' % (0.010, 0.030, 0.100, 0.300, 1.000, 3.000, 10.000, 30.000))
    count = 0
    for i in range(len(param)):
        content = '%.3f\t' % param[i]
        for j in range(len(param)):
            content += '%.3f\t' % acc[count]
            count += 1
        content += '\n'
        f.write(content)
    f.close()

    # results visualizing
    # C = 1
    # sigma = 0.1
    # model = svm.svmTrain_SMO(X, y, C, kernal_function='gaussian', K_matrix=svm.gaussianKernel(X, sigma))
    # print('Results visualizing ...')
    # svm.visualizeBoundaryGaussian(X, y, model, sigma)

def Task2():
    # load data
    print('Loading data ...')
    X_train, y_train = loadData(file_path + '\spamTrain.mat')
    X_test, y_test = loadData(file_path + '\spamTest.mat')
    # plotData(X_train, y_train, title='Task 2')

    print('Program paused. Press enter to continue.\n')
    input()

    # trainging
    svm = SVM()
    print('Number of samples: {}'.format(X_train.shape[0]))
    print('Number of features: {}'.format(X_train.shape[1]))
    print('Training Linear SVM ...')
    C = 1; sigma = 0.01;
    model = svm.svmTrain_SMO(X_train, y_train, C, max_iter=20)
    pred_train = svm.svmPredict(model, np.mat(X_train), sigma)
    acc_train = 1 - abs(np.sum(pred_train - y_train)) / len(y_train)
    print('Train accuracy: {}'.format(acc_train))

    # test
    print('Number of samples: {}'.format(X_test.shape[0]))
    print('Number of features: {}'.format(X_test.shape[1]))
    pred_test = svm.svmPredict(model, np.mat(X_test), sigma)
    acc_test = 1 - abs(np.sum(pred_test - y_test)) / len(y_test)
    print('Test accuracy: {}'.format(acc_test))

if __name__ == '__main__':
    Task1()