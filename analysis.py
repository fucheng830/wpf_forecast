#coding:utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from unitls import *
from statsmodels.graphics import tsaplots

def stationarity_test(df, cap):

    fig = plt.figure(figsize=(12,8))
    p = df['p'].dropna()
    
    ax1 = fig.add_subplot(211)
    
    fig = tsaplots.plot_acf(p, lags=50, ax=ax1)
    
    ax2 = fig.add_subplot(212)
    fig = tsaplots.plot_pacf(p, lags=50, ax=ax2)
    
    """
    ax3 = fig.add_subplot(413)
    fig = tsaplots.plot_acf(df['p'].dropna().diff(1).dropna(), lags=p.shape[0]-2, ax=ax3)
    ax4 = fig.add_subplot(414)
    fig = tsaplots.plot_acf(df[key].dropna().diff(1).dropna().diff(1).dropna(), lags=p.shape[0]-3, ax=ax4)
    """
    plt.show()

def power_plot(df):
    f, ax = plt.subplots(2)
    df[u'实测功率(kW)'].plot(ax=ax[0])
    ax[0].set_title(u'实测功率')
    
    df[u'实测功率(kW)'].hist(color='k', alpha=0.5, bins=50, ax=ax[1])
    ax[1].set_title(u'风功率分布图')
    
    #df[u'实测功率(kW)'].hist(ax=ax[1])
    #ax[1].set_title(u'实测功率')
    plt.show()
    
    
def load_data(file_path):
    df = pd.read_excel(file_path,header=1)
    df = df.iloc[:-2,:]
    df[u'时间'] = pd.to_datetime(df[u'时间'])
    df[u'预测功率(kW)'] = as_float(df[u'预测功率(kW)'])
    df[u'实测功率(kW)'] = as_float(df[u'实测功率(kW)'])
    df = df.set_index(keys=u'时间')
    return df

def as_float(ser):    
    new_data = []
    for r in ser:
        if isinstance(r, unicode):
            r = np.NaN#float(r)
        new_data.append(r) 
    return new_data
    
    
if __name__ == '__main__':
    #df = load_data('data/xts.xlsx')
    #power_plot(df[:8000])
    df = pd.read_csv('data/region0214.csv')
    df[u'实测功率(kW)'] = df['realp']*1000
    df[u'预测工率(kW)'] = df['reportpower']*1000
    df[u'时间'] = df['reporttime1']
    df[u'对数变换风功率'] = GL_transform(one_zero_normal(df[u'实测功率(kW)'], 0 ,99000), 0.00001)
    #power_plot(df[:8000])
    #df[u'对数变换风功率'].hist(color='k', alpha=0.5, bins=50)
    df['p'] = df[u'对数变换风功率']
    #plt.show()
    stationarity_test(df, 99000)