# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:43:15 2022

@author: gojja
"""

import pandas as pd
import numpy as np

x = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\x_df.csv")
y = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\passive_prices_df.csv")

# Creating train, val and test subsamples, for each fund.
def subsample(y, x):
    X = pd.merge(y,x, on=['year', 'month', 'day'], how = "inner")
    X = X.iloc[:,3:]
    subsets = np.zeros(shape=(3,y.iloc[:,3:].shape[1]), dtype=object)
    for i in range(y.iloc[:,3:].shape[1]):
        temp = X.iloc[:, np.r_[i, -11:0]].dropna(how='any')
        obs = len(temp.dropna(how='any'))
        train = temp.sample(n=(round(obs*0.6)),axis=0)
        n_train = temp.drop(train.index.tolist())
        val = n_train.sample(frac = 0.5,axis=0)
        test = n_train.drop(val.index.tolist())
        subsets[0,i], subsets[1,i], subsets[2,i] = train.index.tolist(), val.index.tolist(), test.index.tolist()
    return(subsets)

subsets = subsample(y,x)

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Input
from keras.models import Model, Sequential
import keras
import tensorflow as tf



def scale(y, x):
    X = pd.merge(y,x, on=['year', 'month', 'day'], how = "inner")
    X = X.iloc[:,3:]
    scaler = MinMaxScaler()
    for i in range(y.iloc[:,3:].shape[1]):
        temp = X.iloc[:, np.r_[0, -11:0]].dropna(how='any')
        transf = pd.DataFrame(scaler.fit_transform(temp)).set_index(temp.index)
        X_train = transf.loc[subsets[0,0]]
        X_val = transf.loc[subsets[1,0]]
        X_test = transf.loc[subsets[2,0]]
            
        input1 = Input(shape=11)
        ff = Dense(15, activation = 'relu')(input1)
        ff = Dense(5, activation = 'relu')(ff)
        out = Dense(1, activation = 'relu')(ff)
        model = Model(inputs=input1, outputs=out)
        
        model.compile(loss = 'mean_absolute_error', optimizer = 'adam',metrics=['accuracy'])

        history = model.fit(x=X_train.iloc[:,1:], y=X_train.iloc[:,0], validation_data = (X_val.iloc[:,1:], X_val.iloc[:,0]), epochs = 100)
        
        
    return(transf)
        

# fix the scaling so that we fit on the train and apply on the train, val and test seperatly.









n_fund_p = pd.DataFrame(ind_fund(df))
n_fund_sort = n_fund.sort_values(by=[0])

n_fund.columns = ['values']
n_fund_count = n_fund.assign(more_1000=lambda x: x.values >= 1000)
n_fund_count = n_fund.assign(more_2000=lambda x: x.values >= 2000)
n_fund_count = n_fund.assign(more_3000=lambda x: x.values >= 3000)
n_fund_count = n_fund.assign(more_6000=lambda x: x.values >= 6000)


print(sum(n_fund_count.iloc[:,1]))


n_fund_2 = pd.DataFrame(ind_fund(df_2))

n_fund_2.columns = ['values']
n_fund_2_count = n_fund_2.assign(more_1000=lambda x: x.values >= 1000)
n_fund_2_count = n_fund_2.assign(more_2000=lambda x: x.values >= 2000)
n_fund_2_count = n_fund_2.assign(more_3000=lambda x: x.values >= 3000)
n_fund_2_count = n_fund_2.assign(more_6000=lambda x: x.values >= 6000)


print(sum(n_fund_2_count.iloc[:,1]))


# Randomly assigning train, validation and test dataset index
def r_tvt(n_obs):
    for i in range(n_fund.shape[0]):
        


df_2.index = pd.to_datetime(df_2.index)
df_2["year"], df_2["month"], df_2["day"] = df_2.index.year, df_2.index.month, df_2.index.day
df_2.insert(0, "year", df_2.pop("year"))
df_2.insert(1, "month", df_2.pop("month"))
df_2.insert(2, "day", df_2.pop("day"))




