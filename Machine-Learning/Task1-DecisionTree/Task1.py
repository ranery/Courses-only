# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
from math import log
import operator
import matplotlib
import matplotlib.pyplot as plt

"""
Moudule - Create decision tress for datasets
:createTree(main)
:bestFeature
:calcShannonEnt
:splitDataset
:majorityCnt
"""
def createTree(dataset, label):
    classlist = [example[-1] for example in dataset]
    # no subtree
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    # no feature
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    # choose best feature
    bestFeat = bestFeature(dataset)
    bestFeatLabel = label[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(label[bestFeat])
    # recursion for each subdataset
    feat_values = [example[bestFeat] for example in dataset]
    uniqueVals = set(feat_values)
    for value in uniqueVals:
        sub_label = label[:]
        myTree[bestFeatLabel][value] = createTree(splitDataset(dataset,bestFeat,value), sub_label)
    return myTree

def bestFeature(dataset):
    num_features = len(dataset[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeat = 0
    for i in range(0, num_features):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            sub_dataset = splitDataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(sub_dataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat

def calcShannonEnt(dataset):
    num_samples = len(dataset)
    labelCounts = {}
    for feat_vector in dataset:
        current_label = feat_vector[-1]
        if current_label not in labelCounts.keys():
            labelCounts[current_label] = 0
        labelCounts[current_label] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / num_samples
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataset(dataset, axis, value):
    sub_dataset = []
    for feat_vector in dataset:
        if feat_vector[axis] == value:
            reduce_feat_vector = feat_vector[:axis]
            reduce_feat_vector.extend(feat_vector[axis+1:])
            sub_dataset.append(reduce_feat_vector)
    return sub_dataset

def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

"""
Module - Plot decision tree constructed before
:createPlot(main)
:plotTree
:plotNode
:plotMidText
:getNumLeaves
:getTreeDepth
"""
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def createPlot(myTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=True)
    plotTree.totalW = float(getNumLeaves(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()
    plt.savefig

def plotTree(myTree, parentPt, nodeTxt):
    num_leaves = getNumLeaves(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(num_leaves))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def plotMidText(cntrPt, parentPt, txtString):
    # positon for text
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center")

def getNumLeaves(myTree):
    numLeaves = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeaves += getNumLeaves(secondDict[key])
        else:   numLeaves +=1
    return numLeaves

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

if __name__ == '__main__':
    # load dataset
    f = open('watermelon.txt', 'r', encoding='utf-8')
    label = f.readline()
    label = label.strip().split('  ')
    dataset = [inst.strip().split('  ') for inst in f.readlines()]
    print('feature ', label)
    print('dataset ', dataset)
    # tree structure
    myTree = createTree(dataset, label)
    print('myTree  ', myTree)
    # draw tree
    from pylab import *
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    createPlot(myTree)