#coding:utf-8

import numpy as np
import math
from statsmodels.tsa.api import adfuller
from statsmodels.graphics import tsaplots

def precision(r, p):
    """r实际功率,p预测功率"""
    abs_error = np.abs(r-p)
    avg_error = abs_error/np.sum(abs_error)
    return 1-2*sum(((r/(r+p))-0.5)*avg_error)

def rmse_1(r, p):
    """r实际功率,p预测功率"""
    return 1-np.sqrt(np.sum(np.square((r-p)/r.max()))/r.shape[0])

def uv2sd(u, v):
    pi = np.pi
    s = np.sqrt(u ** 2 + v ** 2)
    d=np.mod(270-np.arctan2(v,u)*180./pi,360.)
    return np.array([s, d]) 

def one_zero_normal(x, xmin, xmax):
    """0-1 power normal"""
    _p_list = []
    for r in x:
        if r > xmax:
            r = 1 
        elif r < xmin:
            r = 0
        else:
            r = r/xmax
        _p_list.append(r)
    return np.array(_p_list)

def GL_transform(x, epsilon):
    """general logarithm transform"""
    boundary = (math.log(epsilon/(1-epsilon)), -math.log(epsilon/(1-epsilon)))
    y_list = []
    for r in x:
        if r>=(1-epsilon):
            y_list.append(boundary[1])
        elif r<=(epsilon):
            y_list.append(boundary[0])
        else:
            #print math.log(float(r)/(1-r))
            y_list.append(math.log(float(r)/(1-r)))
            
    return np.array(y_list)

def y_trans_x(y, epsilon):
    """y transform to x"""
    result = []
    for r in y:
        x = 1/(1+math.exp(-r))
        if x >= (1-epsilon):
            x = 1
        elif x <= epsilon:
            x = 0
        result.append(x)
    return np.array(result)
"""
def rmse(y_test, y):
    """"""
    if isinstance(y_test, list):
        y_test = np.array(y_test)
    if isinstance(y, list):
        y = np.array(y)
    return np.sqrt(np.mean((y_test - y) ** 2))
"""
def rmse(r, p, cap):
    """"""
    
    return 1-np.sqrt(np.sum(np.square((r-p)/cap))/r.shape[0])

def mae(y_test, y):
    """"""
    if isinstance(y_test, list):
        y_test = np.array(y_test)
    if isinstance(y, list):
        y = np.array(y)
    #return np.mean(np.abs(y_test - y)/y)
    data = []
    for a,b in zip(y_test, y):
        if b>0:
            data.append(abs(a-b)/b)
            
    return np.array(data).mean()

def unit_root(ser):
    
    t_boundary = [
                  {'1%':-3.96,'5%':-3.41,'10%':-3.12},
                  {'1%':-3.43,'5%':-2.86,'10%':-2.57},
                  {'1%':-2.58,'5%':-1.95,'10%':-1.61},
                  ]
    """
    t分布临界值(n=∞)
    1%  -2.33
    5%  -1.65
    10% -1.28
    """
    result = adfuller(ser,maxlag=40,store=False,regresults=False)
    #print result
    
    for i,row in enumerate(t_boundary):
        if result[4]['1%'] > row['1%']:
            print "计算式%s,0.01显著水平下t>临界值%s,不能拒绝原假设,存在单位根,时间序列数据不平稳"%(3-i,row['1%'])
        else:
            print "计算式%s,0.01显著水平下t<临界值%s,拒绝原假设,存在单位根,时间序列数据平稳"%(3-i,row['1%'])
            
        if result[4]['5%'] > row['5%']:
            print "计算式%s,0.05显著水平下t>临界值%s,不能拒绝原假设,存在单位根,时间序列数据不平稳"%(3-i,row['5%'])
        else:
            print "计算式%s,0.05显著水平下t<临界值%s,拒绝原假设,存在单位根,时间序列数据平稳"%(3-i,row['5%'])
            
        if result[4]['10%'] > row['10%']:
            print "计算式%s,0.10显著水平下t>临界值%s,不能拒绝原假设,存在单位根,时间序列数据不平稳"%(3-i,row['10%'])
        else:
            print "计算式%s,0.10显著水平下t<临界值%s,拒绝原假设,存在单位根,时间序列数据平稳"%(3-i,row['10%'])
            
    #print "数据取%s阶滞后下，LM检验表明模型残差项不存在自相关性"%result[2]
    print result[4]