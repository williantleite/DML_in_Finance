# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:23:26 2022

@author: gojja och willi
"""

#%% Packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
from sklearn.model_selection import train_test_split

#%% Loading data (induvidual fund)

y_p = pd.read_csv(r".../Data/Clean/passive_returns_m_df.csv", index_col=0)
y_p = y_p.iloc[:,0:3].dropna()

y_a = pd.read_csv(r".../Data/Clean/active_returns_m_df.csv", index_col=0)

#x = pd.read_csv(r".../Data/Clean/x_mon_df.csv")
x = pd.read_csv(r".../Data/Clean/x_df_2.csv")
#x.drop('USREC', inplace=True, axis=1)

#%% Creating X and y (individual fund)

df = pd.merge(y_p,x, on=['year', 'month'], how = "inner")
df = df.assign(day = 1)
df.index = pd.to_datetime(df[['year', 'month', 'day']])
df.drop(['year', 'month', 'day'], inplace=True, axis=1)

y = df.iloc[:,0]
X = df.iloc[:,1:]

#%% Spliting data (induvidual fund)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


#%% Linear regression (induvidual fund)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', model.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test, y_test)))

# plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = -0.05, xmax = 0.05, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()

predictions = model.predict(X_test)
y_bar = y_test.mean()

on = sum((y_test-y_bar)**2)/y_test.shape[0]
vn = sum((y_test-predictions)**2)/y_test.shape[0]
sn = on - vn
r2 = sn/on

# =============================================================================
#%% Aggregate y
# =============================================================================

#Loading data (agg_y)
df = pd.read_csv(r".../Data/Clean/X_time_fix_mon.csv")

#%% Creating X and y (agg_y)
X = df.iloc[:,1:]
y = df.iloc[:,0]

#%% Spliting data (agg_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#%% Linear regression (agg_y)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', model.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test, y_test)))

# plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = -0.05, xmax = 0.05, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()

predictions = model.predict(X_test)
y_bar = y_test.mean()

on = sum((y_test-y_bar)**2)/y_test.shape[0]
vn = sum((y_test-predictions)**2)/y_test.shape[0]
sn = on - vn
r2 = sn/on

# =============================================================================
#%% Regular in loop
# =============================================================================

# Loading data

y_p = pd.read_csv(r".../Data/Clean/passive_returns_m_df.csv", index_col=0)
y_a = pd.read_csv(r".../Data/Clean/active_returns_m_df.csv", index_col=0)

x = pd.read_csv(r".../Data/Clean/x_df_2.csv")

#df = df.assign(day = 1)
#df.index = pd.to_datetime(df[['year', 'month', 'day']])
#df.drop(['year', 'month', 'day'], inplace=True, axis=1)

def num_obs(df):
    obs = np.zeros(shape = (df.shape[1]-2,1))
    for i in range (df.shape[1]-2):
        obs[i] = df.value_counts(subset=df.columns[i+2]).shape[0]
    return(obs)

n_obs_p = pd.DataFrame(num_obs(y_p))
n_obs_p.index = y_p.columns[2:]

n_obs_a = pd.DataFrame(num_obs(y_a))
n_obs_a.index = y_a.columns[2:]

n_obs_few_p = n_obs_p[n_obs_p>=24].dropna()
n_obs_few_a = n_obs_a[n_obs_a>=24].dropna()

sel_p = n_obs_few_p.index
sel_a = n_obs_few_a.index

y_2_p = y_p[sel_p]
y_2_p.insert(0,'month',y_p['month'])
y_2_p.insert(0,'year',y_p['year'])

y_2_a = y_a[sel_a]
y_2_a.insert(0,'month',y_a['month'])
y_2_a.insert(0,'year',y_a['year'])

df = pd.merge(y_2,x, on=['year', 'month'], how = "inner")

def lin_reg(df,y):
    r2 = np.zeros(shape =(y.shape[1]-2,1))
    for i in range(y.shape[1]-2):
        df_temp = df.dropna(axis = 0, how = 'any', subset=df.columns[i+2])
        y_temp = df_temp.iloc[:,i+2]
        X_temp = df_temp.iloc[:,y_2.shape[1]:]
        X_temp.insert(0,'const',1) # Read that adding a constatnt would help, but did not imporve the r2
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.3,
                                                    random_state=1)
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        r2[i] = model.score(X_test, y_test)
    return (r2)

r2_test = lin_reg(df,y_2)
r2_test = pd.DataFrame(r2_test)
r2_test_2 = r2_test[r2_test>=-2000].dropna()
r2_test_3 = r2_test[r2_test>=-1000].dropna()
r2_test_3_avg = r2_test_3.mean()
r2_test_2_avg = r2_test_2.mean()
r2_test_avg = r2_test.mean()

sum(r2_test)

predictions = model.predict(X_test)
y_bar = y_test.mean()

on = sum((y_test-y_bar)**2)/y_test.shape[0]
vn = sum((y_test-predictions)**2)/y_test.shape[0]
sn = on - vn
r2 = sn/on

# regression coefficients
print('Coefficients: ', model.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test, y_test)))

# plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = -0.05, xmax = 0.05, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()

predictions = model.predict(X_test)
y_bar = y_test.mean()

on = sum((y_test-y_bar)**2)/y_test.shape[0]
vn = sum((y_test-predictions)**2)/y_test.shape[0]
sn = on - vn
r2 = sn/on

# =============================================================================
#%% loading data (long format)
# =============================================================================

df_long_p = pd.read_csv(r".../Data/Clean/df_m_long_p.csv")
df_long_a = pd.read_csv(r".../Data/Clean/df_m_long_a.csv")

X_p = df_long_p.iloc[:,4:]
y_p = df_long_p.iloc[:,1]

X_a = df_long_a.iloc[:,4:]
y_a = df_long_a.iloc[:,1]

#%% Spliting data (long format)

X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(X_p, 
                                                                        y_p, 
                                                                        test_size=0.3, 
                                                                        random_state=1)

#%% Linear regression (long df format)

model = linear_model.LinearRegression()
model.fit(X_train_long, y_train_long)

# regression coefficients
print('Coefficients: ', model.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test_long, y_test_long)))

# plot for residual error
plt.style.use('fivethirtyeight')
plt.scatter(model.predict(X_train_long), model.predict(X_train_long) - y_train_long,
            color = "green", s = 10, label = 'Train data')
plt.scatter(model.predict(X_test_long), model.predict(X_test_long) - y_test_long,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = -0.05, xmax = 0.05, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()

predictions_long = model.predict(X_test_long)
y_bar_long = y_test_long.mean()

on_long = sum((y_test_long-y_bar_long)**2)/y_test_long.shape[0]
vn_long = sum((y_test_long-predictions_long)**2)/y_test_long.shape[0]
sn_long = on_long - vn_long
r2_long = sn_long/on_long

# Notes for future research:
# Try the aggregate version
# Try the dis-aggregated version but while not allowing low amounts of obs.
# Try the panel-data version with the correct data format. (normalization?, growth rates) 