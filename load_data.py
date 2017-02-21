#coding:utf-8

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def as_float(ser):    
    new_data = []
    for r in ser:
        if isinstance(r, unicode):
            r = np.NaN#float(r)
        new_data.append(r) 
    return new_data

def read_history_data(file_path):
    df = pd.read_excel(file_path,header=1)
    df = df.iloc[:-2,:]
    df['ptime'] = pd.to_datetime(df[u'时间'])
    df['p'] = as_float(df[u'实测功率(kW)'])
    #df = df.set_index(keys='时间')
    return df
    
def read_gfs(file_path):
    """气象数据拼接.采用根据文件时间延迟12小时拼接
       input : 
             file_path, str :拼接气象数据目录。
             save_path, str :拼接结果保存目录.   
    """      
    file_list = os.listdir(file_path)
    data = []
    for r in file_list:
        #print r,r[-6:]
        #i +=1 
        #if i >10:
            #continue
        if r[-6:] == '12.txt' or r[-6:] == '00.txt':
            #print r,r[-6:]
            dt_time = datetime.strptime('%s %s:00:00'%(r[:-6],r[-6:-4]), '%Y%m%d %H:%M:%S')
            #print r,r[-6:], dt_time
            df = pd.read_table(file_path+r,header=None,encoding='utf-8',delim_whitespace=True,dtype={4:np.str})
            #print df
            if df.shape[1] == 16:
                df.columns=['lon','lat','height','day','time','uspeed','vspeed','tem','rhumidity','pressure','irradiance','surte','cloud_low','m','h','airdensity']
                new_df=df[(df['height']==70)].copy()
                new_df['ptime'] = pd.to_datetime(['%s %s'%(a,b) for a,b in zip(new_df['day'], new_df['time'])], errors='coerce')
                new_df = new_df[(new_df['ptime']>=(dt_time+timedelta(hours=12))) & (new_df['ptime']<(dt_time+timedelta(hours=24)))]
                data.append(new_df)
                #print new_df
    df = pd.concat(data)
    df['ptime'] = df['ptime'].apply(lambda x:x+timedelta(hours=8))
    return df

def merge_data(fpath1, fpath2, save_path, position):
    #measure_data = pd.read_excel(fpath1)
    measure_data = read_history_data(fpath1)
    nwp_data = pd.read_excel(fpath2)
    #print type(nwp_data['lon'][0])
    temp = nwp_data[(nwp_data['lon']==position[0]) & (nwp_data['lat']==position[1])]
    #print measure_data
    #measure_data = measure_data[['ptime', 'p', 'pwspd']]
    new_df = pd.merge(measure_data, temp, left_on='ptime', right_on='ptime')
    new_df = new_df[['ptime','p','']]
    print new_df
    new_df.to_excel(save_path)
if __name__ == '__main__':
    df = read_gfs('data/znxs/')
    df.to_excel('data/znxs_gfs.xlsx')
    merge_data('data/znxs.xlsx', 'data/znxs_gfs.xlsx', 'data/znxs_train.xlsx', (105.61, 37.27))
    
    