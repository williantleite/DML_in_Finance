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

y_p = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Fund data/passive_returns_df.csv")
y_p.index = pd.to_datetime(y_p[['year', 'month', 'day']])
y_p = y_p.iloc[:,0:4].dropna()


x = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/x_df.csv")
x.index = pd.to_datetime(x[['year', 'month', 'day']])
x.drop('recession', inplace=True, axis=1)


df_test = pd.merge(y_p,x, on=['year', 'month', 'day'], how = "inner")
df_test.index = pd.to_datetime(df_test[['year', 'month', 'day']])
df_test.drop(['year', 'month', 'day'], inplace=True, axis=1)








X = x.iloc[:,3:]
X.drop('recession', inplace=True, axis=1)
X.index = pd.to_datetime(x[['year', 'month', 'day']])

#df = pd.DataFrame(x.iloc[:,3])
df = x.iloc[:,4]
df.index = pd.to_datetime(x[['year', 'month', 'day']])

df_multi = x.iloc[:,3:5]
df_multi.index = pd.to_datetime(x[['year', 'month', 'day']])



def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(series, color='red')
    plt.ylabel('Search Frequency for "Heater"', fontsize=16)


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
        print('Normalization complete')
        print(df.columns[i])
        df.iloc[:,i] = DeTrend(df.iloc[:,i])
        print('DeTrend complete')
        print(df.columns[i])
        df.iloc[:,i] = Ch_Vol(df.iloc[:,i])
        print('Ch_Vol complete')
        print(df.columns[i])
        df.iloc[:,i] = Re_Sea(df.iloc[:,i])
        print('Re_Sea complete')
        print(df.columns[i])
        plot_series(df.iloc[:,i])
        plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    return(df)



# Testing on induvidual varible:
df_2 = Normalization(df)

plot_series(df)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

plot_series(df_2)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


df_3 = DeTrend(df_2)

plot_series(df_3)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)


df_4 = Ch_Vol(df_3)

plot_series(df_4)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)

df_5 = Re_Sea(df_4)

plot_series(df_5)
plt.axhline(0, linestyle='--', color='k', alpha=0.3)



# Testing a data frame with multiple vatribles:

    
X_testing = time_fix(df_test).iloc[1:,:]





X_testing = time_fix(X.iloc[:,0:8])
X_testing = time_fix(X.iloc[:,9:11])
X_testing = time_fix(X)
X_testing['year'], X_testing['month'], X_testing['day'] = x['year'], x['month'], x['day']

X_testing = time_fix(df_test)


df_2.index = pd.to_datetime(df_2.index)
X_testing["year"], X_testing["month"], X_testing["day"] = X_testing.index.year, X_testing.index.month, X_testing.index.day





X_testing = pd.merge(y_p,X_testing, on=['year', 'month', 'day'], how = "inner")




X_testing = x
X_testing = time_fix(X_testing.iloc[:,3:])



df_multi_3  = time_fix(df_multi)




plot_pacf(X_testing['consumer_sent'])
plt.show()

X_testting_3 = pd.merge(y_p_1,X_testing, on=['year', 'month', 'day'], how = "inner")
X_testting_4 = X_testting_3.iloc[:,3:]


lags = 14


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

values_test_2 = corr_y_x_lag(X_testing,lags=14)[1:,:]

values_test_3 = pd.DataFrame(values_test_2).astype(float)

p_vals = values_test_3[values_test_3<=0.05]




def corr_y_x_lag(df, lags):
    for i in range(1,df.shape[1]):
        for lag in range(1,lags):
            y = df.iloc[lag:,0]
            x = df.iloc[:-lag,0]
        return [pearsonr(y, x)]





