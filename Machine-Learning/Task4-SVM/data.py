# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    dataDict = loadmat(filename)
    return dataDict['X'], dataDict['y']

def plotData(X, y, title=None):
    X_pos = []
    X_neg = []

    sampleArray = np.concatenate((X, y), axis=1)
    for array in list(sampleArray):
        if array[-1]:
            X_pos.append(array)
        else:
            X_neg.append(array)

    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    if title: ax.set_title(title)

    pos = plt.scatter(X_pos[:, 0], X_pos[:, 1], marker='+', c='b')
    neg = plt.scatter(X_neg[:, 0], X_neg[:, 1], marker='o', c='y')

    plt.legend((pos, neg), ('postive', 'negtive'), loc=2)

    plt.show()
