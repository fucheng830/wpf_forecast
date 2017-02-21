#coding:utf-8

import os
import pandas as pd
import numpy as np
import pickle
from pybrain.tools.shortcuts import buildNetwork  
from pybrain.datasets import SupervisedDataSet  
from pybrain.supervised.trainers import BackpropTrainer  
from pybrain.structure import TanhLayer,LinearLayer,SigmoidLayer,SoftmaxLayer   
from pybrain.tools.validation import CrossValidator,Validator  
import copy
from unitls import *
import matplotlib.pyplot as plt
from tsa_train import back_test


def preprocess(x, y):
    """准备训练数据
    """

    DS = SupervisedDataSet(x.shape[1],1)
    # 往数据集内加样本点
    # 假设x1，x2，x3是输入的三个维度向量，y是输出向量，并且它们的长度相同
    
    for i in range(x.shape[0]):
        DS.addSample(list(x[i,:]), [y[i]])
    
    # 如果要获得里面的输入／输出时，可以用
    #X = DS['input']
    #Y = DS['target']  
    dataTrain = DS
    return dataTrain
    
def creat_model(dataTrain, hidden_layer1, hidden_layer2, xdim, ydim):
    """创建模型
    """
    #隐含层使用的是tanh函数，输出则使用的是y=x线性输出函数  
    ann=buildNetwork(xdim,hidden_layer1,hidden_layer2,ydim,hiddenclass=TanhLayer,outclass=LinearLayer)  
    #BP算法训练，参数为学习率和动量  
    trainer=BackpropTrainer(ann,dataset=dataTrain,learningrate=0.1,momentum=0.1,verbose=True)  
    #trainer.trainEpochs(epochs=20)                         #epochs表示迭代的次数  
    trainer.trainUntilConvergence(maxEpochs=100)             #以上这两种训练方法都可以，看自己喜欢 
    #print CrossValidator(trainer, dataTrain, n_folds=5, max_epochs=20)    #交叉验证 
    return ann
    

def train(x, y, model_path, hidden_layer=(20,5)):
    train_data = preprocess(x, y) 
    model = creat_model(train_data, hidden_layer[0], hidden_layer[1], x.shape[1], 1) #输入输出维度
    f = open(model_path, 'wb')
    pickle.dump(model,f)
    f.close()
    
def ann_predict(test_x, model_path):
    f = open(model_path,'rb')
    model = pickle.load(f)
    # activate函数即神经网络训练后，预测的X2的输出值
    # 可以将其打印出来
    result = []
    for i in range(test_x.shape[0]):
        pre_power = model.activate(test_x[i])
        result.append(pre_power[0])
    return result

def test(dir_name, cap): 
    if not os.path.exists('./result/%s/multiple_pre/'%dir_name):
        os.makedirs('./result/%s/multiple_pre/'%dir_name)
    
    cap = 172500
    df = pd.read_excel('data/%s.xlsx'%dir_name)
    df['theoryp'] = (df['theoryp']*1000).apply(lambda x:1 if x<0 else x)
    df['p'] = (df['p']*1000).apply(lambda x:1 if x<0 else x)
    df = df.dropna()
    df.index = range(df.shape[0])
    print 'start %s, end %s, len %s'%(df['ptime'][0],df['ptime'][df.shape[0]-1],df.shape[0])
    
    data = one_zero_normal(df['p'], 0, cap)
    x = []
    p = []
    for i in range(data.shape[0]):
        if i>5:
            x.append(data[i-5:i])
            p.append(data[i])
    x = np.array(x)
    p = np.array(p)
    train_len = int(data.shape[0]*0.8)
    print 'train len %s, test len %s'%(train_len, data.shape[0]-train_len)
    train(x[:train_len], p[:train_len], 'model/tsa_ann.model', hidden_layer=(20,5))
    pre_y = ann_predict(x[train_len:], 'model/tsa_ann.model')
    print 'rmse: %s'%rmse(np.array(pre_y), np.array(p[train_len:]), cap)
    
    """
    day_rmse = []
    for ptime, grouped in new_result.groupby(lambda x:'%s-%s'%(x.year, x.month)):
        
        for day, day_data in grouped.groupby(lambda x:'%s-%s-%s'%(x.year, x.month, x.day)):
            plt.figure()
            day_data[['ar', 'keep', 'ann', 'multiple', 'p']].plot()
            plt.legend(loc='best')
            ar_rmse = rmse(day_data['ar'],day_data['p'])
            keep_rmse = rmse(day_data['keep'],day_data['p'])
            ann_rmse = rmse(day_data['ann'],day_data['p'])
            multiple_rmse = rmse(day_data['multiple'],day_data['p'])
            
            day_rmse.append([day, ar_rmse, keep_rmse, ann_rmse, multiple_rmse])
            plt.title('ar_rmse:%2.2f, keep_rmse:%2.2f, ann_rmse:%2.2f, multiple_rmse:%2.2f'%(ar_rmse, keep_rmse, ann_rmse, multiple_rmse))
            plt.savefig('./result/%s/multiple_pre/forecast_%s.png'%(dir_name,day))
        
        ar_rmse = rmse(grouped['ar'], grouped['p'])
        keep_rmse = rmse(grouped['keep'], grouped['p'])
        ann_rmse = rmse(grouped['ann'], grouped['p'])
        multiple_rmse = rmse(grouped['multiple'], grouped['p'])
        #theory_rmse = rmse(grouped['theory'], grouped['p'])
        day_rmse.append([ptime, ar_rmse, keep_rmse, ann_rmse, multiple_rmse])
    day_rmse_df = pd.DataFrame(day_rmse,columns=['pectime', 'ar', 'keep', 'ann', 'multiple'])
    day_rmse_df.to_csv('./result/%s/multiple_pre/rmse.csv'%dir_name)
    new_result.to_csv('./result/%s/multiple_pre/forecast_result.csv'%dir_name)
    """
    
if __name__ == '__main__':
    test('nxtx', 172500)
    
    