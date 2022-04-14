# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:37:58 2022

@author: gojja
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr
import numpy as np

# Loading data:

y_p = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Fund data/passive_data_monthly.csv")
y_p.drop('Date', inplace=True, axis=1)
y_p = y_p.iloc[:,0:3].dropna()


x = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/x_mon_df.csv")
x.drop('rec', inplace=True, axis=1)


df = pd.merge(y_p,x, on=['year', 'month'], how = "inner")
df = df.assign(day = 1)
df.index = pd.to_datetime(df[['year', 'month', 'day']])
df.drop(['year', 'month', 'day'], inplace=True, axis=1)



def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(series, color='red')
    plt.ylabel('House Price Index', fontsize=16)


# Normalization:
def Normalization(df):
    avg, dev = df.mean(), df.std()
    df = (df - avg) / dev
    return df



# De-trend (first diff):
def DeTrend(df):
    df = df.diff().dropna()
    return(df)


# Removing increasing volatility:
def Ch_Vol(df):
    annual_volatility = df.groupby(df.index.year).std()
    df_annual_vol = df.index.map(lambda d: annual_volatility.loc[d.year])
    df = df / df_annual_vol
    return(df)




# Removing seasonality:
def Re_Sea(df):
    month_avgs = df.groupby(df.index.month).mean()
    heater_month_avg = df.index.map(lambda d: month_avgs.loc[d.month])
    df = df - heater_month_avg
    return(df)




# Doing it for all varibles in a df:
def time_fix(df):
    for i in range(df.shape[1]):
        df.iloc[:,i] = Normalization(df.iloc[:,i])
        print('Normalization' ,df.columns[i], 'complete')
        df.iloc[:,i] = DeTrend(df.iloc[:,i])
        print('DeTrend', df.columns[i], 'complete')
        df.iloc[:,i] = Ch_Vol(df.iloc[:,i])
        print('Ch_Vol', df.columns[i], 'complete')
        df.iloc[:,i] = Re_Sea(df.iloc[:,i])
        print('Re_Sea', df.columns[i], 'complete')
        plot_series(df.iloc[:,i])
        plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    return(df)



# Testing on induvidual varible:
df_2 = Normalization(df)

plot_series(df.iloc[:,5])
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

plot_series(df_2.iloc[:,5])
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


df_3 = DeTrend(df_2)

plot_series(df_3.iloc[:,5])
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


df_4 = Ch_Vol(df_3)

plot_series(df_4.iloc[:,5])
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

df_5 = Re_Sea(df_4)

plot_series(df_5.iloc[:,5])
plt.axhline(0, linestyle='--', color='k', alpha=0.3)



# Testing a data frame with multiple vatribles:

    
X_time_fix = time_fix(df).iloc[1:,:]



def corr_y_x_lag(df, lags):
    values = np.zeros(shape=(lags,df.iloc[:,1:].shape[1]), dtype=object)
    df_temp = df.iloc[:,1:df.shape[1]]
    for i in range(df_temp.shape[1]):
        for lag in range(1,lags):
            y = df.iloc[lag:,0]
            x = df_temp.iloc[:-lag,i]
            values[lag,i] = pearsonr(y, x)[1]
            print(df.columns[i+1],'Lag: %s'%lag)
            print(values[lag,i])
            print('------')
    return(values)

lags = 30

p_val = corr_y_x_lag(X_time_fix,lags)[1:,:]
p_val = pd.DataFrame(p_val).astype(float)
p_val = p_val[p_val<=0.05]




def corr_y_x_lag(df, lags):
    for i in range(1,df.shape[1]):
        for lag in range(1,lags):
            y = df.iloc[lag:,0]
            x = df.iloc[:-lag,0]
        return [pearsonr(y, x)]




# VAR model:

cols = X_time_fix.columns.tolist()
cols = cols[-10:] + cols[:-10]
X_time_fix = X_time_fix[cols]

model = VAR(X_time_fix)
model_fit = model.fit(maxlags=30)
summ = model_fit.summary()
model_fit.summary()

hjoadfsp =model_fit.get_eq_index('AAINX')

hej = summ._coef_table


VAR()
