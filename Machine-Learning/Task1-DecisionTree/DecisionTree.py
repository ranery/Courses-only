# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:59:48 2018

@author: Administrator
"""
from math import log
import operator
import matplotlib.pyplot as plt
'''
dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
labels = ['no surfacing','flippers']
'''
#导入数据
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']

#计算香农熵，dataSet包括所有实例及其类标签
def calcShannonEnt(dataSet):
    #计算数据集中实例的总数
    numEntries = len(dataSet)
    labelCounts = {}
    #统计类标签及其数量
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

#按照给定特征划分数据集，dataSet为待划分的数据集，
#axis为划分数据集的特征，value为需要返回的特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
    #将符合特征的数据抽取出来
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
 
#获得信息增益最高的特征       
def chooseBestFeatureToSplit(dataSet):
    #特征数量 
    numFeatures = len(dataSet[0]) - 1  
    #划分前数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures): 
    #类标签列表
        featList = [example[i] for example in dataSet]
    #类标签并集
        uniqueVals = set(featList)       
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
    #计算按照第i个特征划分后的香农熵
            newEntropy += prob * calcShannonEnt(subDataSet)  
    #计算信息增益
        infoGain = baseEntropy - newEntropy 
    #选择信息增益最大的特征
        if (infoGain > bestInfoGain):       
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature 
        
#若所有特征被遍历完而集合中类别标签不唯一，则采用多数表决方法决定分类标签。
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] # the difference with py3.x

#在该代码中，若特征被遍历完，则停止遍历
#创建树的函数，labels为特征名称
def createTree(dataSet,labels):
    #生成类别标签列表
    classList = [example[-1] for example in dataSet]
    #如果类别标签属于同一类，则停止创建
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    #如果没有特征，则采用多数表决方法决定分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最好的特征标号
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #删除已参与划分的特征
    del(labels[bestFeat])
    #得到该特征包含的所有值列表
    featValues = [example[bestFeat] for example in dataSet]
    #求取该特征所有值的并集
    uniqueVals = set(featValues)
    for value in uniqueVals:
    #复制所有的特征名称
        subLabels = labels[:]       
    #递归调用createTree函数创建树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

#获取叶节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
    #判断子节点是否为字典类型，如果是则递归调用函数getNumLeafs（）
    #否则为叶子节点
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
    #判断子节点是否为字典类型，如果是则递归调用函数getTreeDepth（）
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

#decisionNode，leafNode，arrow_args分别定义非叶子节点，叶子节点和箭头类型
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#绘出结点及相应箭头
#输入的参数分别为要显示的文本，文本中心点坐标，指向文本的点坐标，结点类型
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )

#在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    #定义文本信息的位置
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center")    
    
def plotTree(myTree, parentPt, nodeTxt):
    #计算树的宽度和高度
    numLeafs = getNumLeafs(myTree)  
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    #结点文本中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    #父子结点间文本信息
    plotMidText(cntrPt, parentPt, nodeTxt)
    #绘出结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    #根据树词典逐渐绘叶子结点与非叶子结点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict': 
            plotTree(secondDict[key],cntrPt,str(key))        
        else: 
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


#myTree为以构建好用词典类型表示的数
def createPlot(myTree):
    #定义画布，并令其背景为白色
    fig = plt.figure(1, facecolor='white')
    #把画布清空
    fig.clf()
    #横纵坐标轴
    axprops = dict(xticks=[], yticks=[])
    #定义绘图，111表示图的排列，有1行1列的第1个图，frameon表是是否定义矩阵坐标轴；
    createPlot.ax1 = plt.subplot(111, frameon=True)   
    #全局变量，保存树的宽度和深度，方便布局结点位置
    plotTree.totalW = float(getNumLeafs(myTree))
    plotTree.totalD = float(getTreeDepth(myTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    #调用函数plotTree，进行绘图
    plotTree(myTree, (0.5,1.0), '')
    plt.show()
 
if __name__ == '__main__':
    lensesTree = createTree(lenses, lensesLabels)
    createPlot(lensesTree)

    