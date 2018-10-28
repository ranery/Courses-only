# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:20:16 2018

@author: Administrator
"""

import pandas as pd
from numpy import *

dataSetTrain=pd.read_csv('D:\python\TrainLogistic1.csv',encoding='gbk')
dataSetTest=pd.read_csv('D:\python\TestLogistic.csv',encoding='gbk')
'''
#求取每个训练指纹
def ObtainTrainRSSValue(dataSet):
    #初始化矩阵用于存放每个指纹向量
    BSSIDUnion=set(dataSet['BSSIDLabel'])
    listBSSIDUnion=list(BSSIDUnion)
    a = []
    for i in range(0, dataSet.iloc[len(dataSet)-1,4]):
        tmp = []
        for j in range(0, len(BSSIDUnion)):
            tmp.append(0)
        a.append(tmp)
    RSSValue=array(a)
    Labels=[0]*dataSet.iloc[len(dataSet)-1,4]
    p=0
    q=0
    for i in range(1,len(dataSet)):
        if i<len(dataSet)-1:
            if ((dataSet.iat[i-1,2]==dataSet.iat[i,2]) & (dataSet.iat[i-1,4]==dataSet.iat[i,4])):
                p=p
            else :
                q=i
                tempBSSID=dataSet.iloc[p:q,0]
                listBSSID=list(tempBSSID)
                for BSSID in tempBSSID:
                    if BSSID in listBSSID:
                        num1=listBSSID.index(BSSID)+p
                    if BSSID in listBSSIDUnion:
                        num2=listBSSIDUnion.index(BSSID)
                    RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]
                    Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
                p=q    
        elif i==(len(dataSet)-1):
            #print(i)
            tempBSSID=dataSet.iloc[p:i+1,0]
            listBSSID=list(tempBSSID)
            for BSSID in tempBSSID:
                if BSSID in listBSSID:
                    num1=listBSSID.index(BSSID)+p
                if BSSID in listBSSIDUnion:
                    num2=listBSSIDUnion.index(BSSID)
                RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]
                Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
    return RSSValue,Labels,BSSIDUnion

#求取每个测试指纹，与训练指纹不同在于需要以训练指纹的BSSID集合进行展开
def ObtainTestRSSValue(dataSet,BSSIDUnion):
    #初始化矩阵用于存放每个指纹向量
    #BSSIDUnion=set(dataSet['BSSIDLabel'])
    listBSSIDUnion=list(BSSIDUnion)
    a = []
    for i in range(0, dataSet.iloc[len(dataSet)-1,4]):
        tmp = []
        for j in range(0, len(BSSIDUnion)):
            tmp.append(0)
        a.append(tmp)
    RSSValue=array(a)
    Labels=[0]*dataSet.iloc[len(dataSet)-1,4]
    p=0
    q=0
    for i in range(1,len(dataSet)):
        if i<len(dataSet)-1:
            if ((dataSet.iat[i-1,2]==dataSet.iat[i,2]) & (dataSet.iat[i-1,4]==dataSet.iat[i,4])):
                p=p
            else :
                q=i
                tempBSSID=dataSet.iloc[p:q,0]
                listBSSID=list(tempBSSID)
                for BSSID in tempBSSID:
                    if BSSID in listBSSIDUnion:
                        num2=listBSSIDUnion.index(BSSID) 
                        num1=listBSSID.index(BSSID)+p 
                        RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]
                        Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
                p=q    
        elif i==(len(dataSet)-1):
            #print(i)
            tempBSSID=dataSet.iloc[p:i+1,0]
            listBSSID=list(tempBSSID)
            for BSSID in tempBSSID:  
                if BSSID in listBSSIDUnion:
                    num2=listBSSIDUnion.index(BSSID)
                    num1=listBSSID.index(BSSID)+p
                    RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]
                    Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
    return RSSValue,Labels
'''
'''
BSSIDTrain=set(dataSetTest['BSSIDLabel'])
BSSIDTest=set(dataSetTrain['BSSIDLabel'])
BSSIDUnion=BSSIDTrain & BSSIDTrain
'''
'''
#特征选择测试集与训练集的交集
def ObtainTestRSSValue(dataSet,BSSIDUnion):
    #初始化矩阵用于存放每个指纹向量
    #BSSIDUnion=set(dataSet['BSSIDLabel'])
    listBSSIDUnion=list(BSSIDUnion)
    a = []
    for i in range(0, dataSet.iloc[len(dataSet)-1,4]):
        tmp = []
        for j in range(0, len(BSSIDUnion)):
            tmp.append(0)
        a.append(tmp)
    RSSValue=array(a)
    Labels=[0]*dataSet.iloc[len(dataSet)-1,4]
    p=0
    q=0
    for i in range(1,len(dataSet)):
        if i<len(dataSet)-1:
            if ((dataSet.iat[i-1,2]==dataSet.iat[i,2]) & (dataSet.iat[i-1,4]==dataSet.iat[i,4])):
                p=p
            else :
                q=i
                tempBSSID=dataSet.iloc[p:q,0]
                listBSSID=list(tempBSSID)
                for BSSID in tempBSSID:
                    if BSSID in listBSSIDUnion:
                        num2=listBSSIDUnion.index(BSSID) 
                        num1=listBSSID.index(BSSID)+p 
                        RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]
                        Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
                p=q    
        elif i==(len(dataSet)-1):
            #print(i)
            tempBSSID=dataSet.iloc[p:i+1,0]
            listBSSID=list(tempBSSID)
            for BSSID in tempBSSID:  
                if BSSID in listBSSIDUnion:
                    num2=listBSSIDUnion.index(BSSID)
                    num1=listBSSID.index(BSSID)+p
                    RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]
                    Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
    return RSSValue,Labels
'''
#选择AP出现频率该与1-k的AP
def ObtainTrainBSSIDList(dataSet,k):   
    #初始化矩阵用于存放每个指纹向量
    BSSIDUnion=[]
    BSSIDUnion1=set(dataSet['BSSIDLabel'])
    listBSSIDUnion1=list(BSSIDUnion1)
    a = []
    for i in range(0, dataSet.iloc[len(dataSet)-1,4]):
        tmp = []
        for j in range(0, len(BSSIDUnion1)):
            tmp.append(0)
        a.append(tmp)
    RSSValue=array(a)
    Labels=[0]*dataSet.iloc[len(dataSet)-1,4]
    p=0
    q=0
    for i in range(1,len(dataSet)):
        if i<len(dataSet)-1:
            if ((dataSet.iat[i-1,2]==dataSet.iat[i,2]) & (dataSet.iat[i-1,4]==dataSet.iat[i,4])):
                p=p
            else :
                q=i
                tempBSSID=dataSet.iloc[p:q,0]
                listBSSID=list(tempBSSID)
                for BSSID in tempBSSID:
                    if BSSID in listBSSIDUnion1:
                        num2=listBSSIDUnion1.index(BSSID) 
                        num1=listBSSID.index(BSSID)+p
                        RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]+100#更改值
                        Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
                p=q    
        elif i==(len(dataSet)-1):
            #print(i)
            tempBSSID=dataSet.iloc[p:i+1,0]
            listBSSID=list(tempBSSID)
            for BSSID in tempBSSID:  
                if BSSID in listBSSIDUnion1:
                    num2=listBSSIDUnion1.index(BSSID)
                    num1=listBSSID.index(BSSID)+p
                    RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]+100#更改值
                    Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]
    for j in range(shape(RSSValue)[1]):
        num=0
        for i in range(shape(RSSValue)[0]):
            if RSSValue[i,j]==0:
                num+=1
        if num/(shape(RSSValue)[0])<k:
           BSSIDUnion.append(listBSSIDUnion1[j])
    return BSSIDUnion
BSSIDUnion=ObtainTrainBSSIDList(dataSetTrain,0.95)

#特征选择，将含有值的作为1，不含有值的作为0,或用100+原有值
def ObtainRSSValue01(dataSet,BSSIDUnion):
    #初始化矩阵用于存放每个指纹向量
    #BSSIDUnion=set(dataSet['BSSIDLabel'])
    listBSSIDUnion=list(BSSIDUnion)
    a = []
    for i in range(0, dataSet.iloc[len(dataSet)-1,4]):
        tmp = []
        for j in range(0, len(BSSIDUnion)):
            tmp.append(0)
        a.append(tmp)
    RSSValue=array(a)
    Labels=[0]*dataSet.iloc[len(dataSet)-1,4]
    p=0
    q=0
    for i in range(1,len(dataSet)):
        if i<len(dataSet)-1:
            if ((dataSet.iat[i-1,2]==dataSet.iat[i,2]) & (dataSet.iat[i-1,4]==dataSet.iat[i,4])):
                p=p
            else :
                q=i
                tempBSSID=dataSet.iloc[p:q,0]
                listBSSID=list(tempBSSID)
                for BSSID in tempBSSID:
                    if BSSID in listBSSIDUnion:
                        num2=listBSSIDUnion.index(BSSID) 
                        num1=listBSSID.index(BSSID)+p
                        RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]+100#更改值
                        Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]-1
                p=q    
        elif i==(len(dataSet)-1):
            #print(i)
            tempBSSID=dataSet.iloc[p:i+1,0]
            listBSSID=list(tempBSSID)
            for BSSID in tempBSSID:  
                if BSSID in listBSSIDUnion:
                    num2=listBSSIDUnion.index(BSSID)
                    num1=listBSSID.index(BSSID)+p
                    RSSValue[dataSet.iloc[num1,4]-1,num2]=dataSet.iloc[num1,1]+100#更改值
                    Labels[dataSet.iloc[num1,4]-1]=dataSet.iloc[num1,2]-1
    return RSSValue,Labels

def sigmoid(inZ):
    return 1.0/(1+exp(-inZ))

#改进的随机梯度算法，dataMatrix为样本数组，
#classLabels为类别标签，numIter为迭代次数
def stocGradAscent(RSSValue, Labels, numIter):
    m,n = shape(RSSValue)
    #初始化系数为1
    weights = ones(n)   
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    
    #从数据集中随机选择一个样本        
            randIndex = int(random.uniform(0,len(dataIndex)))
    #利用现有系数计算sigmoid函数值
            h = sigmoid(sum(RSSValue[randIndex]*weights))
    #标签类别与sigmoid函数值的差值
            error = Labels[randIndex] - h
    #更新权重
            weights = weights + alpha * error * RSSValue[randIndex]
    #从数据集列表中删除该样本
            del(dataIndex[randIndex])
    return weights

#预测测试样本的分类标签，inX为测试样本，
#weights为训练得到的系数
def classifyVector(inX, weights):
    #调用sigmoid计算sigmoid值，大于0.5时，预测为1；否则预测为0.
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    
def colicTest(dataSetTrain, dataSetTest,BSSIDUnion):    
#def colicTest(dataSetTrain, dataSetTest):
    #RSSValueTrain,LabelsTrain,BSSIDUnion=ObtainTrainRSSValue(dataSetTrain)
    RSSValueTrain,LabelsTrain=ObtainRSSValue01(dataSetTrain,BSSIDUnion)
    RSSValueTest,LabelsTest=ObtainRSSValue01(dataSetTest,BSSIDUnion)
    #调用改进的随机梯度上升算法训练得到系数 
    trainWeights = stocGradAscent(RSSValueTrain, LabelsTrain, 10)
    #初始化预测错误样本数量与总样本数量
    errorCount = 0
    numTestVec =shape(RSSValueTest)[0]
    #逐行读取训练数据，并按照制表符进行分割
    for i in range(numTestVec):
        if int (classifyVector(RSSValueTest[i], trainWeights))!=int(LabelsTest[i]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print ("the error rate of this test is: %f" % errorRate)
    return errorRate