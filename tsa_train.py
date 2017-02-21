#coding:utf-8

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import AR,VAR,ARMA,ARIMA
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
from statsmodels.tsa.tsatools import unintegrate,unintegrate_levels
import matplotlib.pyplot as plt
import math
import os
import copy
from unitls import *


def back_test(data, start_index=50000, train_len=3000, p=4, steps=16):
    pre_data = []
    for x in xrange(start_index,data.shape[0]-steps):
        rw = AR(data[x-train_len:x]).fit(p)
        ar_pre = _ar_predict_out_of_sample(data, np.array(copy.deepcopy(rw.params)), p, 1, steps, start=x)
        keep_pre = keep_predict(data, x, train_len)
        pre_data.append([data.index[x+steps-1], ar_pre[steps-1], keep_pre[steps-1]])

    result = pd.DataFrame(pre_data,columns=['ptime','ar','keep'])
    result['ptime'] = pd.to_datetime(result['ptime'])
    result = result.set_index(keys='ptime')
    return result

def _ar_predict_out_of_sample(y, params, p, k_trend, steps, start=0):
    mu = params[:k_trend] or 0  # only have to worry about constant
    arparams = params[k_trend:][::-1]  # reverse for dot
    # dynamic endogenous variable
    endog = np.zeros(p + steps)  # this is one too big but doesn't matter
    if start:
        endog[:p] = y[start-p:start]
    else:
        endog[:p] = y[-p:]

    forecast = np.zeros(steps)
    for i in range(steps):
        fcast = mu + np.dot(arparams, endog[i:i+p])
        forecast[i] = fcast
        endog[i + p] = fcast

    return list(forecast)

def _arma_predict(rw, data, steps, order):
    #print rw.resid
    return  _arma_predict_out_of_sample(np.array(rw.params), steps, np.array(rw.resid), order[0], order[1], rw.k_trend, rw.k_exog,
                                data, exog=None, start=0, method='mle')
   
def _arima_predict(rw, data, steps, order):
    d = order[1]
    _endog = np.diff(data, n=d)
    forecast = _arma_predict_out_of_sample(np.array(rw.params), steps, np.array(rw.resid),
                                               order[0], order[2],
                                               rw.k_trend, rw.k_exog,
                                               _endog,
                                               exog=None, method='css-mle')

    
    endog = data[-d:]
    forecast = unintegrate(forecast, unintegrate_levels(endog, d))[d:]
    return forecast
    
def keep_predict(ser, x, train_len):
    predict_ser = [ser[x-1] for i in range(16)]
    return predict_ser
    
if __name__ == '__main__':
    pass
    #df = pd.read_csv('result/dgl_new/dgl_new_data.csv')
    #print precision(df['p'],df['ar'])
    #print rmse_1(df['p'],df['ar'])
    
    cap = 172500.0
    dir_name = 'nxtx'
    if not os.path.exists('./result/%s'%dir_name):
        os.makedirs('./result/%s'%dir_name)
    df = pd.read_excel('data/%s.xlsx'%dir_name)
    df = df.set_index(keys='ptime',drop=False)
    #print pd.isnull(df)
    print df.shape
    df = df.dropna()
    print df.shape
    df['theoryp'] = (df['p']*1000).apply(lambda x:1 if x<0 else x)
    df['p'] = (df['p']*1000).apply(lambda x:1 if x<0 else x)
    df['p1'] = one_zero_normal(df['theoryp'], 0, cap)
    #df['p2'] = one_zero_normal(df['p']*1000, 0, cap)
    step_result = []
    for i in range(16):
        data = []
        result_df = back_test(df['p1'],start_index=3000, train_len=3000, p=10, steps=i+1)
        result_df['p'] = df['p']
        result_df['ar'] = result_df['ar']*cap.apply(lambda x:0 if x<0 else x)
        result_df['keep'] = result_df['keep']*cap
        result_df.to_csv('./result/%s/%s_%s_forecast.csv'%(dir_name, dir_name,i+1))
        #按天画预测图
        for ptime, grouped in result_df.groupby(lambda x:'%s-%s-%s'%(x.year, x.month, x.day)):
            #plt.figure()
            #grouped[['ar','keep','p']].plot()
            #plt.legend(loc='best')
            #plt.title('%s ar_rmse:%2.2f, keep_rmse:%2.2f,'%(ptime, rmse(grouped['ar'],grouped['p'])/cap, rmse(grouped['keep'],grouped['p'])/cap))
            #plt.savefig('./result/%s/%s.png'%(dir_name, ptime))
            data.append([ptime,rmse(grouped['ar'], grouped['p'], cap), precision(grouped['ar'], grouped['p']), rmse_1(grouped['ar'],grouped['p'])])
        
        new_df = pd.DataFrame(data,columns=[u'ptime',u'rmse',u'precision',u'rmse_1'])
        new_df['ptime'] = pd.to_datetime(new_df['ptime'])
        step_result.append([i+1,new_df['rmse'].mean(),new_df['precision'].mean(),new_df['rmse_1'].mean()])
        new_df = new_df.sort_values(by='ptime')
        new_df.to_excel('result/%s/%s.xlsx'%(dir_name, i+1))
    df_avg = pd.DataFrame(step_result, columns=[u'step',u'rmse',u'precision',u'rmse_1'])
    df_avg.to_excel('result/%s/dgl_result.xlsx'%dir_name)
        
    