# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:43:15 2022

@author: gojja och willi
"""

import pandas as pd
import numpy as np

x = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\x_df.csv")
y = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\passive_prices_df.csv", index_col=0)

agg_y = y.iloc[:,:3]
agg_y["mean"] = y.iloc[:,3:].mean(axis=1)
agg_y = agg_y[np.isfinite(agg_y).all(axis = 1)]

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

subsets = subsample(agg_y,x)

from sklearn.preprocessing import StandardScaler

from keras.layers import Dense, Input
from keras.models import Model, Sequential
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

def scale(y, x):
    X = pd.merge(agg_y,x, on=['year', 'month', 'day'], how = "inner")
    X = X.iloc[:,3:]
    scaler = StandardScaler()
    for i in range(y.iloc[:,3:].shape[1]):
        temp = X.dropna(how='any')
        X_train = temp.loc[subsets[0,0]]
        X_val = temp.loc[subsets[1,0]]
        X_test = temp.loc[subsets[2,0]]
        X_train = pd.DataFrame(scaler.fit_transform(X_train)).set_index(X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val)).set_index(X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test)).set_index(X_test.index)
        
        input1 = Input(shape=11)
        ff = Dense(250, activation = 'tanh')(input1)
        ff = Dense(200, activation = 'tanh')(ff)
        ff = Dense(150, activation = 'tanh')(ff)
        ff = Dense(100, activation = 'tanh')(ff)
        ff = Dense(50, activation = 'tanh')(ff)
        out = Dense(1, activation = 'tanh')(ff)
        model = Model(inputs=input1, outputs=out)
        
        model.compile(loss = ['mse'], optimizer = 'adam')

        history = model.fit(x=X_train.iloc[:,1:], y=X_train.iloc[:,0], validation_data = (X_val.iloc[:,1:], X_val.iloc[:,0]), epochs = 1000)
        
        test_loss = model.evaluate(x = X_test.iloc[:,1:], y = X_test.iloc[:,0])
        
        y_hat = pd.DataFrame(model.predict(X_test.iloc[:,1:]))
        
    return(transf)

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Loss')
    plt.plot(hist['epoch'], hist['val_loss'],
             label = 'Val Loss')
    plt.ylim([0.5,1.2])
    plt.legend()
    plt.figure()
    plt.show()

plot_history(history)
        

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




