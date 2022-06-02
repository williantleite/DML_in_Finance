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
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
    df = df.set_index("DATE").resample("M").ffill()
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
#    df.iloc[:,2] = df.iloc[:,2].diff()
#    df.iloc[:,2] = Ch_Vol(df.iloc[:,2])
#    df.iloc[:,2] = De_Sea(df.iloc[:,2])
    df["year"] = df["year"].astype(str)
    df['month'] = df['month'].astype(str)
    df["DATE"] = df[["year", "month"]].agg("-".join, axis=1)
    df = pad_monthly(df)
    df = df.dropna()
    return df

def transform_hpi(df):
    df = adding_date_variables(df)
    df.iloc[:,2] = Normalization(df.iloc[:,2]) #Normalize
    df.iloc[:,2] = df.iloc[:,2].diff()
#    df.iloc[:,2] = Ch_Vol(df.iloc[:,2])
#    df.iloc[:,2] = De_Sea(df.iloc[:,2])
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
#        df.iloc[:,i] = DeTrend(df.iloc[:,i])
#        print('DeTrend', df.columns[i], 'complete')
#        df.iloc[:,i] = Ch_Vol(df.iloc[:,i])
#        print('Ch_Vol', df.columns[i], 'complete')
#        df.iloc[:,i] = De_Sea(df.iloc[:,i])
#        print('De_Sea', df.columns[i], 'complete')
        plot_series(df.iloc[:,i])
        plt.axhline(0, linestyle='--', color='k', alpha=0.3)
    return(df)

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

def corr_y_x_lag(df, lags):
    for i in range(1,df.shape[1]):
        for lag in range(1,lags):
            y = df.iloc[lag:,0]
            x = df.iloc[:-lag,0]
        return [pearsonr(y, x)]
    
def adf_test(df):
    print("")
    print ('Results of Dickey-Fuller Test: %s' %(df.name))
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

#%% Load data

y_p = pd.read_csv(r"F:\Thesis\Data\Clean\passive_prices_m_df.csv", index_col=0)
y_a = pd.read_csv(r"F:\Thesis\Data\Clean\active_prices_m_df.csv", index_col=0)

agg_y = y_p.iloc[:,:2]
agg_y["mean"] = y_p.iloc[:,2:].mean(axis=1)
agg_y = agg_y[np.isfinite(agg_y).all(axis = 1)]

agg_a = y_a.iloc[:,:2]
agg_a['mean'] = y_a.iloc[:,2:].mean(axis=1)
agg_a = agg_a[np.isfinite(agg_a).all(axis=1)]

x = pd.read_csv(r"F:\Thesis\Data\Clean\x_df.csv")
recession = x.iloc[:,:2]
recession['recession'] = x.pop('recession')

x.loc[:,'consumer_sent'] = x.loc[:,'consumer_sent'].pct_change(periods=1)
x.loc[:,'inflation'] = x.loc[:,'inflation'].pct_change(periods=1)
x.loc[:,'m2'] = x.loc[:,'m2'].pct_change(periods=1)
x.loc[:,'hpi'] = x.loc[:,'hpi'].pct_change(periods=1)
x = x.dropna()

for i in range(x.shape[1]):
    adf_test(x.iloc[:,i])

x.pop('nrou')
x.pop('interest_rate')

#%% Fix the quarterly variables:

anxious_index_df = pd.read_excel(r"F:\Thesis\Data\Raw Data\Other Variables\Anxious Index\anxious_index_chart.xlsx")
anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
anxious_index_df = anxious_index_df.astype({"anxious_index": "float64"})
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = anxious_index_df.drop(["year", "quarter", "month"], axis = 1)

gdp_df = pd.read_csv(r"F:\Thesis\Data\Raw Data\Other Variables\Real-GDP\Real_GDP.csv")
gdp_df.iloc[:,1] = gdp_df.iloc[:,1].pct_change(periods=1)

house_price_index_df = pd.read_csv(r"F:\Thesis\Data\Raw Data\Other Variables\House prices\All-Transactions_House_Price_Index.csv")
house_price_index_df.iloc[:,1] = house_price_index_df.iloc[:,1].pct_change(periods=1)

anxious_index_df = transform_pad(anxious_index_df)
gdp_df = transform_pad(gdp_df)
hpi_df = transform_hpi(house_price_index_df)

#%% Update X

df = pd.merge(agg_a,x, on=['year', 'month'], how = "inner")
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

# df_4 = Ch_Vol(df_3.loc[:, 'nrou'])

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
                  recession]

X_time_fix = reduce(lambda left,right: pd.merge(left, right, on=['year', 'month'], how = "inner"), variables_list)

X_time_fix.insert(3, "anxious_index_y", X_time_fix.pop('anxious_index_y'))
X_time_fix.pop('anxious_index_x')
X_time_fix.insert(6, "GDPC1", X_time_fix.pop("GDPC1"))
X_time_fix.pop('gdp_growth')
X_time_fix.insert(7, "USSTHPI", X_time_fix.pop("USSTHPI"))
X_time_fix.pop('hpi')
X_time_fix.pop('year')
X_time_fix.pop('month')

X_time_fix = X_time_fix.rename({'anxious_index_y' : 'anxious_index', 
                                'GDPC1' : 'gdp_growth',
                                'USSTHPI' : 'hpi_growth'}, axis = 1)

for i in range(X_time_fix.shape[1]):
    adf_test(X_time_fix.iloc[:,i])

#%% VAR model:
    
#%% Split in train-test using leave-one-out method

tscv = TimeSeriesSplit(n_splits=424, test_size=1)
forecast = pd.DataFrame()
X_train_list = []
X_test_list = []
for train_index, test_index in tscv.split(X_time_fix):
    X_train_list.append([train_index])
    X_test_list.append([test_index])
    X_train, X_test = X_time_fix.iloc[train_index], X_time_fix.iloc[test_index]
    model = VAR(X_train)
    model_fit = model.fit(7)    
    laged_values = X_train.values[-7:]
    forecast_temp = pd.Series(model_fit.forecast(y = laged_values, steps=1)[0][0],
                              name = X_test.index.values.astype(int)[0])
    forecast = forecast.append(forecast_temp)

cols = X_time_fix.columns.tolist()
cols = cols[-10:] + cols[:-10]
X_time_fix = X_time_fix[cols]

lags = 24

model = VAR(X_time_fix)
model_fit = model.fit(maxlags=lags)
summ = model_fit.summary()
coef_table = summ._coef_table()

coef_df = np.zeros(shape=((1+X_time_fix.shape[1]*lags),5), dtype=object)

for i in range(1+X_time_fix.shape[1]*lags):
    coef_df[i,0] = coef_table[281+85*i:306+85*i].strip() #Name
    coef_df[i,1] = coef_table[307+85*i:320+85*i] #Coeff
    coef_df[i,2] = coef_table[321+85*i:335+85*i] #Std.error
    coef_df[i,3] = coef_table[340+85*i:355+85*i] #t-stat
    coef_df[i,4] = coef_table[356+85*i:366+85*i] #prob (p-val)

coef_df = pd.DataFrame(coef_df, columns = ['var','coeff','std. err', 't-stat', 'p-val'])

for i in range(1,coef_df.shape[1]):
    coef_df = coef_df.astype({coef_df.iloc[:,i].name:'float'})

coef_df_sig = coef_df.loc[coef_df['p-val'] <= 0.05, ['var','coeff','std. err', 't-stat', 'p-val']]
coef_df_sig['lag'] = coef_df_sig.iloc[:,0].str.extract('(\d+)')
coef_df_sig.loc[1:,'var'] = coef_df_sig.iloc[1:,0].str.split('.').str[1]

for i in range(X_time_fix.shape[1]):
    print(X_time_fix.columns[i])
    perform_adf_test(X_time_fix.iloc[:,i])

for i in range(X_time_fix.shape[1]):
    plot_acf(X_time_fix.iloc[:,i])
    plt.title(X_time_fix.columns[i], fontsize=16)
    plt.show()
    
#%% Forecast tryout:
    
#First we define the optimal number of legs using AIC

results_aic = []
for p in range(1,24):
  results = model.fit(p)
  results_aic.append(results.aic)

results_bic = []
for p in range(1,24):
  results = model.fit(p)
  results_bic.append(results.bic)
  
sns.set()
plt.plot(list(np.arange(1,24,1)), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC")
plt.show()

sns.set()
plt.plot(list(np.arange(1,24,1)), results_bic)
plt.xlabel("Order")
plt.ylabel("BIC")
plt.show()

# Summaries
fit_aic = model.fit(7)
fit_bic = model.fit(1)
fit_aic.summary()
fit_bic.summary()

# Forecast

laged_values_aic = X_time_fix.iloc[:424,:].values[-7:]
forecast_aic = pd.DataFrame(fit_aic.forecast(y = laged_values_aic, steps=10))
forecast_aic

forecast_aic_u = (forecast_aic.iloc[:,0] + agg_y.iloc[:,2].mean())*agg_y.iloc[:,2].std()

#%% Plot detrended variables

fig, axs = plt.subplots(2, 4, figsize=(24, 12))
axe = axs.ravel()
for i in range(X_time_fix.iloc[:,1:9].shape[1]):
    X_time_fix.iloc[:,(1+i)].plot(ax=axe[i], title=X_time_fix.iloc[:,(1+i)].name)
fig.suptitle('Model predictors (post processing)', fontsize=16, y=.93)
plt.show()
