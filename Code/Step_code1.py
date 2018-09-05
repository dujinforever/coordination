# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 08:50:40 2018

@author: Dujin
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from scipy import interpolate
import heapq
import math
import matplotlib.pyplot as plt
from scipy import signal
import os
import sklearn.cluster
import sys
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
'''constName_joint =['腰椎屈曲,deg', 'Lumbar Lateral - RT,deg', 'Lumbar Axial - RT,deg',
       '胸椎屈曲,deg', 'Thoracic Lateral - RT,deg', 'Thoracic Axial - RT,deg',
       '肘关节屈曲 左,deg', '肘关节屈曲 右,deg', 
			'肩屈曲 左,deg',
       '肩屈曲 右,deg', '肩外展 左,deg', '肩外展 右,deg', 'Shoulder Rotation - out 左,deg',
       'Shoulder Rotation - out 右,deg', '伸腕 左,deg', '伸腕 右,deg', '手腕径向 左,deg',
       '手腕径向 右,deg', 'Wrist Supination 左,deg', 'Wrist Supination 右,deg',
       '髋关节屈曲 左,deg', '髋关节屈曲 右,deg', '髋关节外展 左,deg', '髋关节外展 右,deg',
       'Hip Rotation - out 左,deg', 'Hip Rotation - out 右,deg', '膝关节屈曲 左,deg',
       '膝关节屈曲 右,deg', '踝关节背屈 左,deg', '踝关节背屈 右,deg', '踝关节反转 左,deg',
       '踝关节反转 右,deg', '踝关节外展 左,deg', '踝关节外展 右,deg']'''
#该函数用以得出周期的索引值，以波谷为起始点，找寻下一个波谷的位置
def loaddata(folder_name,type = 'np'):
    cwd = os.getcwd()
    folder_name = cwd+'\\'+folder_name
    files = os.listdir(folder_name)
    conbine_list = []
    for file in files:
        if not os.path.isdir(file):
            if type == 'np':
                Data = np.loadtxt(folder_name +'\\'+file,dtype=float,delimiter =',')
                conbine_list.append((file,Data))
            if type == 'pd':
                Data = pd.read_csv(folder_name+'\\'+file)
                conbine_list.append((file,Data))
    return conbine_list
def Buttterworth(X,fs,fc,types):
    X = np.array(X)
    fc = fc/(np.array(fs)/2)
    b,a = signal.butter(4,fc,btype = types)
    Y = []
    try:
        for i in range(X.shape[1]):
            zi = signal.lfilter_zi(b, a)
            z, _ = signal.lfilter(b, a, X[:,i], zi=zi*X[0,i])
            Y.append(z)
    except:
            zi = signal.lfilter_zi(b, a)
            z, _ = signal.lfilter(b, a, X, zi=zi*X[0])
            Y.append(z)
    Y = np.array(Y)
    Y = np.array(Y)
    Y = Y.transpose()
    return Y
def CycleIndex(data, n_Marker = 50, n_clusters = 7,head_nMarker=300):
    list_data = list(data)
    head = heapq.nlargest(head_nMarker,enumerate(list_data), key=lambda x: x[1])
    h = pd.DataFrame(head)
    startIndex = min(h[0])
    endIndex = max(h[0])
    list_data1 = list_data[startIndex:endIndex]
    cc = heapq.nsmallest(n_Marker, enumerate(list_data1), key=lambda x: x[1]) #get the list_type
    h = pd.DataFrame(cc)
    Index = h[0]+startIndex
    Value = h[1]
    h = pd.concat([Index,Value],axis=1)
    plt.scatter(h[0],h[1])
    plt.plot(list_data)
    km = KMeans(n_clusters)
    km.fit(h)
    center = km.cluster_centers_
    Index = np.floor(np.sort(center[:,0]))
    return Index

def CycleInterp(X,n_Marker = 100):
    size = X.shape
    x = np.arange(0,size[0])
    f = interpolate.interp1d(x, X)
    xNew = np.linspace(1,size[0]-1,num = n_Marker)
    Y = f(xNew)
    return Y

def Cycle_n(DataFrame,Index):
    #dividing the a long series of data including multiple cycle to  single cycle data accoring cycle and properties  
    DataFrameList = []
    n = len(Index)
    shape = DataFrame.shape
    dictA ={}
    for i in range(shape[1]):
        dictA[DataFrame.columns[i]] = []
    NonCycleDataFrame = DataFrame.iloc[int(np.min(Index)):int(np.max(Index)),:]
    for i in range(n-1):
        single_DataFrame = DataFrame.iloc[int(Index[i]):int(Index[i+1]),:]
        newDataFrame = pd.DataFrame()
        for i in range(shape[1]):
            temp = single_DataFrame[single_DataFrame.columns[i]]
            InterpValue = CycleInterp(temp)
           # plt.figure(i)
           # plt.plot(InterpValue)
           #plt.title(single_DataFrame.columns[i])
            dictA[single_DataFrame.columns[i]].append(InterpValue)
            newDataFrame[single_DataFrame.columns[i]] = InterpValue    
        DataFrameList.append(newDataFrame)
    for key,values in dictA.items():
        dictA[key] = pd.DataFrame(values).T
    return DataFrameList,dictA,NonCycleDataFrame
	
def phase_normal_theta(theta):
	theta = 2*(theta - theta.min())/(theta.max() - theta.min())-1
	return theta
	
def phase_noraml_omega(theta):
	omega = theta.diff()
	omega = np.nan_to_num(omega)
	omega = 2*(omega - omega.min())/(omega.max() - omega.min())-1
	return omega
	
def cal_PhaseAngle(DataFranme):
	X = DataFranme.apply(phase_angle,axis = 0)
	X = np.delete(np.array(X),0,0)
	Mean = np.mean(X,axis = 1)
	return (X,Mean)
import scipy.stats as stats

def jointrandomness(Y,Mean,type = ['corr','rms']):
	corr = np.array([])
	std = np.array([])
	for i in range(Y.shape[1]):
		corr = np.append(corr,stats.pearsonr(Y[:,i],Mean)[0])
		std = np.append(std,np.std(np.abs(Y[:,i]-Mean)))
	return (corr.mean(),std.mean(),corr)
#相位角
def phase_angle(theta,plt_bool = False):
	omega = theta.diff()
	omega = np.nan_to_num(omega)
	#omega = 2*(omega - omega.min())/(omega.max() - omega.min())-1
	omega = (omega)/(np.abs(omega).max())
	theta = 2*(theta - theta.min())/(theta.max() - theta.min())-1
	phi = np.empty_like(theta)
	for x in range(theta.size):
		i = theta[x]
		j = omega[x]
		if i>=0 and j>=0:
			phi[x] = (np.arctan(j/i)*57.3)
		if i<0 and j>0:
			phi[x] = (180+np.arctan(j/i)*57.3)
		if i>=0 and j<0:
			phi[x] = (np.abs(np.arctan(j/i)*57.3))
		if i<0 and j<0:
			phi[x] = (180-np.arctan(j/i)*57.3)
	phi[np.isnan(phi)] = 0
	if plt_bool == True:
		plt.figure(1)
		plt.scatter(theta,omega)
		plt.annotate('s',xy=(theta[0],omega[0]),xytext=(theta[0],omega[0]))
		plt.figure(2)
		plt.plot(omega)
	return phi
	
def cal_Coordination(table1,table2):
	corr_CRP = np.array([])
	relative_table = table1-table2
	Mean_CRP = relative_table.mean(axis = 1)
	Variability_CRP = relative_table.std(axis = 1)
	for i in range(relative_table.shape[1]):
		corr_CRP = np.append(corr_CRP,stats.pearsonr(relative_table[:,i],Mean_CRP)[0])
	return (Mean_CRP,Variability_CRP,corr_CRP)
	
def intra_coordination(table1,table2):
	table1_PhaseAngle = cal_PhaseAngle(table1)
	table2_PhaseAngle = cal_PhaseAngle(table2)
	result = cal_Coordination(table1_PhaseAngle[0],table2_PhaseAngle[0])
	return result
	
def Compare_Mul_Locomotion(datalist,name1,name2):
	allresult = []
	for i in range(len(datalist)):
		table1 = datalist[i][name1]
		table2 = datalist[i][name2]
		result = intra_coordination(table1,table2)
		allresult.append(result)
	output1 = []
	output2 = []
	output3 = []
	for Mean_CRP,Variability_CRP,corr_CRP in allresult:
		output1.append(Mean_CRP)
		output2.append(Variability_CRP)
		output3.append(corr_CRP)
	return (output1,output2,output3)
	
def FulFilter(X,fs,fc,types):
    X = np.array(X)
    fc = fc/(np.array(fs)/2)
    b,a = signal.butter(4,fc,btype = types)
    Y = []
    try:
        for i in range(X.shape[1]):
            z= signal.filtfilt(b, a, X[:,i])
            Y.append(z)
    except:
            z = signal.filtfilt(b, a, X)
            Y.append(z)
    Y = np.array(Y)
    Y = np.array(Y)
    Y = Y.transpose()
    return Y