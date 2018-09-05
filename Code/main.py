import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import sklearn.cluster
import sys
import pandas as pd


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

def rectifier(X):
    X = np.absolute(X)
    return X


def getmav(X,winsize,wininc):
    size = X.shape
    maxnum_feature = int (np.floor((size[0]-winsize)/wininc))
    X1 = np.zeros((1,size[1]))
    for i in range(maxnum_feature):
        data = np.mean(np.abs(X[wininc*i:wininc*i+winsize]),axis = 0)
        X1  = np.concatenate((X-1,data[np.newaxis,:]),axis = 0)
    X1 = X1[2:,:]
    return X1

def getrms(X,winsize,wininc):
    size = X.shape
    maxnum_feature = int (np.floor((size[0]-winsize)/wininc))
    X1 = np.zeros((1,size[1]))
    for i in range(maxnum_feature):
        data = np.sqrt(np.mean(np.square(X[wininc*i:wininc*i+winsize]),axis = 0))
        X1  = np.concatenate((X1,data[np.newaxis,:]),axis = 0)
    X1 = X1[1:,:]
    return X1

def segment(Y,width):
    size = Y.shape
    maxnum_feature = int (np.floor(size[0]/width))
    Y1 = np.zeros((1,1))
    for i in range(maxnum_feature):
        data = np.array([np.mean(np.abs(Y[width*i:width*(i+1)-1]),axis = 0)])
        Y1  = np.concatenate((Y1,data[np.newaxis,:]),axis = 0)
    Y1 = Y1[1:,:]
    return Y1

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
        
def getdata(folderPath):
    files = os.listdir(folderPath)
    fileName = []
    for file in files:
        if not os.path.isdir(file):
            f1 = open(folderPath +'\\'+file)
            fileName.append(file) 
            for i in range(6):
                line = f1.readline()
            list1 = list(line.strip().split(','))
            data_line = list(map(float, list1))
            table = np.array(data_line)
            table = table[np.newaxis,:];
            line = f1.readline()
            while line!='\n':
                list1 = list(line.strip().split(','))
                data_line = list(map(float, list1))
                data_line = np.array(data_line)
                data_line = data_line[np.newaxis,:]
                table = np.concatenate((table,data_line))
                line = f1.readline()
            conbine_list = []
            conbine_list.append((file,table))
    return conbine_list

def fre_analysis(x,sampling_rate = 1000):
    length = np.array(x).shape[0]
    fft_size = length
    t = np.arange(0,length/sampling_rate,1.0/sampling_rate)
    x_analysis = x[:fft_size]
    xf = np.fft.rfft(x_analysis)/fft_size
    freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
    xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.subplot(211)
    plt.plot(t[:fft_size], x)
    plt.xlabel(u"时间(秒)")
    plt.title(u"Original—Signal波形和频谱")    
    plt.subplot(212)
    plt.plot(freqs, xfp)
    return freqs, xfp
'''#get the file path
    filepath = os.getcwd()
    path = filepath+'\\data'
    filesname = os.listdir(path)
    
    open(path+'\\'+filename)
    
#get the single file data
name = str()
singleData = np.loadtxt(the_path,dtype = float ,delimiter=None)
singlename = name.strip('')
#preprocess
#filtering
X = Buttterworth(singleData,fs,fc,type)
X = getmav(X,winsize,wininc)


if __name__ == '__main__':
    conbine_list = getdata('C:\\Users\\Dujin\\Desktop\\Python\\data')
'''