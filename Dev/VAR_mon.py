# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:37:58 2022

@author: gojja och willi
"""

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy.stats import pearsonr
import numpy as np
from functools import reduce

#%% Set Functions

def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(series, color='red')
    plt.title(series.name, fontsize=16)

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
    annual_vol = df.groupby(df.index.year).std()
    df_annual = df.index.map(lambda d: annual_vol.loc[d.year])
    df = df / df_annual
    return(df)

# Removing seasonality:
def De_Sea(df):
    month_avgs = df.groupby(df.index.month).mean()
    heater_month_avg = df.index.map(lambda d: month_avgs.loc[d.month])
    df = df - heater_month_avg
    return(df)

def pad_monthly(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df = df.set_index("DATE").resample("M").pad()
    df["year"], df["month"] = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

def adding_date_variables(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    df["year"], df["month"], = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)
    
def transform_pad(df):
    df = adding_date_variables(df)
    df.iloc[:,2] = Normalization(df.iloc[:,2]) #Normalize
    df.iloc[:,2] = df.iloc[:,2].diff()
    df.iloc[:,2] = Ch_Vol(df.iloc[:,2])
    df.iloc[:,2] = De_Sea(df.iloc[:,2])
    df["year"] = df["year"].astype(str)
    df['month'] = df['month'].astype(str)
    df["DATE"] = df[["year", "month"]].agg("-".join, axis=1)
    df = pad_monthly(df)
    df = df.dropna()
    return df

# Doing it for all varibles in a df:
def time_fix(df):
    for i in range(df.shape[1]):
        df.iloc[:,i] = Normalization(df.iloc[:,i])
        print('Normalization' ,df.columns[i], 'complete')
        df.iloc[:,i] = DeTrend(df.iloc[:,i])
        print('DeTrend', df.columns[i], 'complete')
        df.iloc[:,i] = Ch_Vol(df.iloc[:,i])
        print('Ch_Vol', df.columns[i], 'complete')
        df.iloc[:,i] = De_Sea(df.iloc[:,i])
        print('De_Sea', df.columns[i], 'complete')
        plot_series(df.iloc[:,i])
        plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    return(df)

#%% Load data

y_p = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\passive_prices_m_df.csv", index_col=0)
y_p = y_p.iloc[:,0:3].dropna()

x = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\x_df.csv")
recession = x.pop('recession')

#%% Fix the quarterly variables:

anxious_index_df = pd.read_excel(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Anxious Index\anxious_index_chart.xlsx")
anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
anxious_index_df = anxious_index_df.astype({"anxious_index": "float64"})
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = anxious_index_df.drop(["year", "quarter", "month"], axis = 1)

gdp_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Real-GDP\Real_GDP.csv")
gdp_df.iloc[:,1] = gdp_df.iloc[:,1].pct_change(periods=1)

nrou_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\Noncyclical_Rate_of_Unemployment.csv")
nrou_df.iloc[:,1] = nrou_df.iloc[:,1].pct_change(periods=1)

house_price_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\House prices\All-Transactions_House_Price_Index.csv")
house_price_index_df.iloc[:,1] = house_price_index_df.iloc[:,1].pct_change(periods=1)

anxious_index_df = transform_pad(anxious_index_df)
gdp_df = transform_pad(gdp_df)
nrou_df = transform_pad(nrou_df)
hpi_df = transform_pad(house_price_index_df)

#%% Update X

df = pd.merge(y_p,x, on=['year', 'month'], how = "inner")
df = df.assign(day = 1)
df.index = pd.to_datetime(df[['year', 'month', 'day']])
df.drop(['year', 'month', 'day'], inplace=True, axis=1)

# Testing on induvidual variable:
# df_2 = Normalization(df)

# plot_series(df.iloc[:,5])
# plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# plot_series(df_2.iloc[:,5])
# plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# df_3 = DeTrend(df_2)

# plot_series(df_3.iloc[:,5])
# plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# df_4 = Ch_Vol(df_3.iloc[:, 5])

# plot_series(df_4)
# plt.axhline(0, linestyle='--', color='k', alpha=0.3)

# df_5 = De_Sea(df_4)

# plot_series(df_5)
# plt.axhline(0, linestyle='--', color='k', alpha=0.3)

#%% Testing a data frame with multiple vatribles:
   
X_time_fix = time_fix(df).iloc[1:,:]
X_time_fix['year'], X_time_fix['month'] = X_time_fix.index.year, X_time_fix.index.month
X_time_fix.insert(0, "year", X_time_fix.pop('year'))
X_time_fix.insert(1, "month", X_time_fix.pop("month"))

variables_list = [X_time_fix,
                  anxious_index_df, 
                  gdp_df, 
                  hpi_df, 
                  nrou_df]

X_time_fix = reduce(lambda left,right: pd.merge(left, right, on=['year', 'month'], how = "inner"), variables_list)

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

lags = 24

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
model_fit = model.fit(maxlags=24)
summ = model_fit.summary()
model_fit.summary()

hjoadfsp =model_fit.get_eq_index('AAINX')

hej = summ._coef_table

VAR()
