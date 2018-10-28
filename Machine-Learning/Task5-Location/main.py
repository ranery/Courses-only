# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
import csv
from data import *
import matplotlib.pyplot as plt

# load data
train_feat, train_x, train_y, val_feat, val_x, val_y, test_feat = dataset()

# train
def calDistance(x, y):
    return np.sqrt(np.square(x) + np.square(y))
def method(model_x, model_y):
    model_x.fit(train_feat, train_x)
    score = model_x.score(val_feat, val_x)
    result_val_x = model_x.predict(val_feat)
    result_test_x = model_x.predict(test_feat)
    print('score of val_x : ', score)
    model_y.fit(train_feat, train_y)
    score = model_y.score(val_feat, val_y)
    result_val_y = model_y.predict(val_feat)
    result_test_y = model_y.predict(test_feat)
    print('score of val_y : ', score)
    print('average deviation of val_x : ', np.average(abs(val_x - result_val_x)))
    print('average deviation of val_y : ', np.average(abs(val_y - result_val_y)))
    print('average deviation of val distance : ', np.average(calDistance(val_x - result_val_x, val_y - result_val_y)))
    if os.path.exists('result.csv'):
        os.remove('result.csv')
    f = open('result.csv', 'a', newline='')
    csv_write = csv.writer(f, dialect='excel')
    for i in range(len(result_test_x)):
        result = []
        result.append(i)
        result.append(result_test_x[i])
        result.append(result_test_y[i])
        csv_write.writerow(result)

    # plot val result figure
    plt.figure(1)
    plt.subplot(131)
    plt.plot(train_x, train_y, 'ro', label='real')
    plt.title('trainset distribution')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(loc='upper right', ncol=1)
    plt.subplot(132)
    plt.plot(val_x, val_y, 'ro', label='real')
    plt.plot(result_val_x, result_val_y, 'bo', label='predict')
    plt.title('valset distribution')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(loc='upper right', ncol=1)
    plt.subplot(133)
    plt.plot(result_test_x, result_test_y, 'bo', label='predict')
    plt.title('testset distribution')
    plt.xlabel('x'); plt.ylabel('y')
    plt.legend(loc='upper right', ncol=1)
    plt.show()

def run(type):
    if type == 'decision_tree':
        from sklearn import tree
        model = tree.DecisionTreeRegressor()
    elif type == 'linear':
        from sklearn import linear_model
        model = linear_model.LinearRegression()
    elif type == 'svm':
        from sklearn import svm
        model = svm.SVR()
    elif type == 'KNN':
        from sklearn import neighbors
        model = neighbors.KNeighborsRegressor()
    elif type == 'random_forest':
        from sklearn import ensemble
        model = ensemble.RandomForestRegressor(n_estimators=20)
    elif type == 'adaboost':
        from sklearn import ensemble
        model = ensemble.AdaBoostRegressor(n_estimators=50)
    elif type == 'extra_tree':
        from sklearn.tree import ExtraTreeRegressor
        model = ExtraTreeRegressor()
    method(model, model)

run('KNN')
