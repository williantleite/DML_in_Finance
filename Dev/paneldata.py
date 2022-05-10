# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:07:22 2022

@author: gojja och willi
"""

# =============================================================================
#%% Packages
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from functools import reduce
from sklearn import linear_model
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import time
from joblib import Parallel
import os

# =============================================================================
#%% Functions
# =============================================================================

def plot_series(series):
    plt.figure(figsize=(12,6))
    plt.plot(series, color='red')
    plt.title(series.name, fontsize=16)

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
    # df.iloc[:,2] = Normalization(df.iloc[:,2]) #Normalize
    # df.iloc[:,2] = df.iloc[:,2].diff()
    # df.iloc[:,2] = Ch_Vol(df.iloc[:,2])
    # df.iloc[:,2] = De_Sea(df.iloc[:,2])
    df["year"] = df["year"].astype(str)
    df['month'] = df['month'].astype(str)
    df["DATE"] = df[["year", "month"]].agg("-".join, axis=1)
    df = pad_monthly(df)
    df = df.dropna()
    return df

def groupaverage(X, G):
    omega_hat = pd.DataFrame(0, index=X.iloc[:,11:20].columns, columns=G)
    for g in range(len(G)):
        omega_hat[G.iloc[g]] = X.loc[X.variable.str.contains(G.iloc[g]), X.iloc[:,11:20].columns].mean()
    omega_hat.index += '_mean'
    omega_hat = omega_hat.T
    return omega_hat

def num_obs(df):
    obs = np.zeros(shape = (df.shape[1]-2,1))
    for i in range (df.shape[1]-2):
        obs[i] = df.value_counts(subset=df.columns[i+2]).shape[0]
    return(obs)

def lagy(X, G):
    omega_hat = pd.DataFrame(0, index=X.index, columns=["date", "variable", "value"])
    omega_hat.iloc[:,0:3] = X.iloc[:, 0:3]
    for g in range(len(G)):
        for i in range(1,8):
            omega_hat["lag_{}".format(i)] = X.groupby('variable').value.shift(i)
    return omega_hat

# =============================================================================
#%% Loading the data
# =============================================================================

y_p = pd.read_csv(r"F:\Thesis\Data\Clean\passive_prices_m_df.csv", index_col=0)
y_a = pd.read_csv(r"F:\Thesis\Data\Clean\active_prices_m_df.csv", index_col=0)

x = pd.read_csv(r"F:\Thesis\Data\Clean\x_df.csv")

# =============================================================================
#%% Data transformation and creation of the X_df
# =============================================================================

# interest_rate = x.iloc[:,:2]
# interest_rate['interest_rate'] = x.pop('interest_rate')
nrou = x.iloc[:,:2]
nrou['nrou'] = x.pop('nrou')
recession = x.iloc[:,:2]
recession['recession'] = x.pop('recession')

x.loc[:,'consumer_sent'] = x.loc[:,'consumer_sent'].pct_change(periods=1)
x.loc[:,'inflation'] = x.loc[:,'inflation'].pct_change(periods=1)
x.loc[:,'m2'] = x.loc[:,'m2'].pct_change(periods=1)
x.loc[:,'hpi'] = x.loc[:,'hpi'].pct_change(periods=1)
x.loc[:,'rou'] = x.loc[:,'rou'].pct_change(periods=1)
x.loc[:,'anxious_index'] = x.loc[:,'anxious_index'].pct_change(periods=1)

# x.loc[:,'nrou'] = x.loc[:,'nrou'].pct_change(periods=1)

anxious_index_df = pd.read_excel(r"F:\Thesis\Data\Raw Data\Other Variables\Anxious Index\anxious_index_chart.xlsx")
anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
anxious_index_df = anxious_index_df.astype({"anxious_index": "float64"})
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = anxious_index_df.drop(["year", "quarter", "month"], axis = 1)
anxious_index_df.iloc[:,0] = anxious_index_df.iloc[:,0].pct_change(periods=1)

gdp_df = pd.read_csv(r"F:\Thesis\Data\Raw Data\Other Variables\Real-GDP\Real_GDP.csv")
gdp_df.iloc[:,1] = gdp_df.iloc[:,1].pct_change(periods=1)

house_price_index_df = pd.read_csv(r"F:\Thesis\Data\Raw Data\Other Variables\House prices\All-Transactions_House_Price_Index.csv")
house_price_index_df.iloc[:,1] = house_price_index_df.iloc[:,1].pct_change(periods=1)

anxious_index_df = transform_pad(anxious_index_df)
gdp_df = transform_pad(gdp_df)
hpi_df = transform_pad(house_price_index_df)

variables_list = [x,
                  anxious_index_df, 
                  gdp_df,
                  hpi_df,
                  recession]

x = reduce(lambda left,right: pd.merge(left, right, on=['year', 'month'], how = "inner"), variables_list)

x.insert(3, "anxious_index_y", x.pop('anxious_index_y'))
x.pop('anxious_index_x')
x.insert(6, "GDPC1", x.pop("GDPC1"))
x.pop('gdp_growth')
x.insert(7, "USSTHPI", x.pop("USSTHPI"))
x.pop('hpi')
# x.pop('year')
# x.pop('month')

x = x.rename({'anxious_index_y' : 'anxious_index', 
              'GDPC1' : 'gdp_growth',
              'USSTHPI' : 'hpi_growth'}, axis = 1)

cols=["year","month"]
x['date'] = x[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
x.insert(2, "date", x.pop("date"))

# =============================================================================
#%% Adding lagged variables
# =============================================================================

x_full = x.copy()

for i in range(1,8):
    x_lag = x.copy()
    x_lag.iloc[:,3:]
    x_lag.iloc[:,3:] = x.iloc[:,3:].shift(periods=i)
    for j in range(x.shape[1]-3):
        x_lag.rename(columns = {x_lag.columns[3+j]:str(x_lag.columns[3+j]+'lag'+ str(i))},inplace=True)
    x_lag = x_lag.iloc[:,2:]
    x_full = pd.merge(x_full,x_lag, on=['date'], how = "inner")

x = x_full.dropna()

# =============================================================================
#%% Dropping funds with less than 8 obs
# =============================================================================

obs_p = pd.DataFrame(num_obs(y_p))
obs_a = pd.DataFrame(num_obs(y_a))

obs_p.index = y_p.columns[2:]
obs_a.index = y_a.columns[2:]

obs_few_p = obs_p[obs_p>=8].dropna()
obs_few_a = obs_a[obs_a>=8].dropna()

y_few_p = y_p[obs_few_p.index]
y_few_p.insert(0,'month',y_p['month'])
y_few_p.insert(0,'year',y_p['year'])
y_p = y_few_p

y_few_a = y_a[obs_few_a.index]
y_few_a.insert(0,'month',y_a['month'])
y_few_a.insert(0,'year',y_a['year'])
y_a = y_few_a

# =============================================================================
#%% Creating of the panel_df
# =============================================================================

y_p['date'] = y_p[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
y_a['date'] = y_a[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")

variables_test_p = list(y_p.columns)
variables_test_p.remove('year')
variables_test_p.remove('month')

variables_test_a = list(y_a.columns)
variables_test_a.remove('year')
variables_test_a.remove('month')

y_long_p=pd.melt(y_p,id_vars='date',value_vars=variables_test_p)
y_long_a=pd.melt(y_a,id_vars='date',value_vars=variables_test_a)

# =============================================================================
#%% Lagged y
# =============================================================================
g_p = y_long_p.loc[:,'variable'].drop_duplicates()
g_a = y_long_a.loc[:, 'variable'].drop_duplicates()

lag_y_p = lagy(y_long_p,g_p)
lag_y_a = lagy(y_long_a,g_a)

lag_y_p = lag_y_p.dropna()
lag_y_a = lag_y_a.dropna()

# =============================================================================
#%% Creating of the panel_df pt2
# =============================================================================

df_long_p = pd.merge(y_long_p,x, on=['date'], how = "inner")
df_long_a = pd.merge(y_long_a,x, on=['date'], how = "inner")

df_long_p = df_long_p.dropna(axis=0,how='any')
df_long_a = df_long_a.dropna(axis=0,how='any')

df_long_p = df_long_p[~df_long_p.isin([np.inf, -np.inf]).any(1)]
df_long_a = df_long_a[~df_long_a.isin([np.inf, -np.inf]).any(1)]

df_long_p = df_long_p.drop(['date'],axis=1)
df_long_a = df_long_a.drop('date',axis=1)

# =============================================================================
#%% Adding means encoding of original X
# =============================================================================

df_long_p_avg = groupaverage(df_long_p, g_p)
df_long_a_avg = groupaverage(df_long_a, g_a)

df_long_p_enc = pd.merge(df_long_p, df_long_p_avg, on=['variable'], how='inner')
df_long_a_enc = pd.merge(df_long_a, df_long_a_avg, on=['variable'], how='inner')

# df_long_p_enc.to_csv("df_long_p_enc.csv", index=True)
# df_long_a_enc.to_csv("df_long_a_enc.csv", index=True)

df_long_p_enc = pd.read_csv(r"F:\Thesis\Data\Clean\df_long_p_enc_df.csv", index_col=0)
df_long_a_enc = pd.read_csv(r"F:\Thesis\Data\Clean\df_long_a_enc_df.csv", index_col=0)

# =============================================================================
#%% Tuning parameters for XGBoost 
# =============================================================================

#%% Y-task
### Passive

X_p = df_long_p_enc.iloc[:,4:]
y_p = df_long_p_enc.iloc[:, 3]
xgb_param = {'learning_rate' : [0.2], 
             'objective' : ['reg:squarederror'], 
             'n_estimators' : [200,500,1000],
             'reg_lambda' : [0,1,2],
             'subsample' : [0.3,0.6,0.9],
             'colsample_bytree' : [0.3,0.6,0.9],
             'max_depth' : [2,6,10]}

# mse = make_scorer(mean_squared_error, greater_is_better = False)

xgb_tune_cv_y_p = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                             xgb_param, 
                             scoring=['neg_mean_squared_error', 'r2'], 
                             # scoring = mse,
                             cv=2, 
                             refit='r2', 
                             verbose = 3,
                             n_jobs=-1,
                             pre_dispatch=2)#turning to +1 lets you see at what iteration it is currently, but using -1 allows the use of all cores. 

# os.environ['JOBLIB_TEMP_FOLDER'] = r'F:\Temp' #necessary for being able to handle the large intermediate results. IF this doesnt work try pathing it to F

a = time.time()
# with Parallel(max_nbytes=None): #Due to limitations in joblib::Parallels we need to use this fix so that the calculations do not consume all available disk memory. This was not necessary once I pathed the intermediate results to disk that had 500gb free. If you dont have it you should use the Parallel() trick.
xgb_tune_cv_y_p.fit(X_p, y_p)
time_cv_y_p = time.time()-a

best_params_y_p = xgb_tune_cv_y_p.best_params_

print('R2: ', r2_score(y_pred = xgb_tune_cv_y_p.predict(X_p), y_true = y_p))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_y_p.predict(X_p), y_true = y_p))

xgb_param_eta_y_p = {'learning_rate' : np.arange(0.01, 0.201, 0.01), 
                     'objective' : ['reg:squarederror'], 
                     'n_estimators' : [best_params_y_p['n_estimators']],
                     'reg_lambda' : [best_params_y_p['reg_lambda']],
                     'subsample' : [best_params_y_p['subsample']],
                     'colsample_bytree' : [best_params_y_p['colsample_bytree']],
                     'max_depth' : [best_params_y_p['max_depth']]}

xgb_tune_cv_eta_y_p = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                                   xgb_param_eta_y_p, 
                                   scoring=['neg_mean_squared_error', 'r2'], 
                                   # scoring = mse,
                                   cv=2, 
                                   refit='r2', 
                                   verbose = 0,
                                   n_jobs=-1,
                                   pre_dispatch=2) 

a = time.time()
xgb_tune_cv_eta_y_p.fit(X_p, y_p)
time_cv_eta_y_p = time.time() - a

best_params_eta_y_p = xgb_tune_cv_eta_y_p.best_params_

print('R2: ', r2_score(y_pred = xgb_tune_cv_eta_y_p.predict(X_p), y_true = y_p))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_eta_y_p.predict(X_p), y_true = y_p))

np.save("best_params_eta_y_p.npy", best_params_eta_y_p)
test = np.load('best_params_eta_y_p.npy',allow_pickle='TRUE').item()

#%% Y-task
### Active

X_a = df_long_a_enc.iloc[:,4:]
y_a = df_long_a_enc.iloc[:,3]

xgb_param = {'learning_rate' : [0.2], 
             'objective' : ['reg:squarederror'], 
             'n_estimators' : [200],
             'reg_lambda' : [0,1,2],
             'subsample' : [0.3,0.6,0.9],
             'colsample_bytree' : [0.3,0.6,0.9],
             'max_depth' : [2,6,10]}

xgb_tune_cv_y_a = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                               xgb_param, 
                               scoring=['neg_mean_squared_error', 'r2'], 
                               # scoring = mse,
                               cv=2, 
                               refit='r2', 
                               verbose = 1,
                               n_jobs=-1) 

a = time.time()
# with Parallel(max_nbytes=None):
xgb_tune_cv_y_a.fit(X_a, y_a) 
time_cv_y_a = time.time()-a

best_params_y_a = xgb_tune_cv_y_a.best_params_

print('R2: ', r2_score(y_pred = xgb_tune_cv_y_a.predict(X_a), y_true = y_a))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_y_a.predict(X_a), y_true = y_a))

xgb_param_eta_y_a = {'learning_rate' : np.arange(0.01, 0.201, 0.01), 
                     'objective' : ['reg:squarederror'], 
                     'n_estimators' : [best_params_y_a['n_estimators']],
                     'reg_lambda' : [best_params_y_a['reg_lambda']],
                     'subsample' : [best_params_y_a['subsample']],
                     'colsample_bytree' : [best_params_y_a['colsample_bytree']],
                     'max_depth' : [best_params_y_a['max_depth']]}

xgb_tune_cv_eta_y_a = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                                   xgb_param_eta_y_a, 
                                   scoring=['neg_mean_squared_error', 'r2'], 
                                   # scoring = mse,
                                   cv=2, 
                                   refit='r2', 
                                   verbose = 1,
                                   n_jobs=-1,
                                   pre_dispatch = 2) 

a = time.time()
xgb_tune_cv_eta_y_a.fit(X_a, y_a)
time_cv_eta_y_a = time.time()-a

best_params_eta_y_a = xgb_tune_cv_eta_y_a.best_params_

print('R2: ', r2_score(y_pred = xgb_tune_cv_eta_y_a.predict(X_a), y_true = y_a))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_eta_y_a.predict(X_a), y_true = y_a))

np.save("best_params_eta_y_a.npy", best_params_eta_y_a)

#%% D-task
### Passive
df_long_p_enc = pd.read_csv(r"F:\Thesis\Data\Clean\df_long_p_enc_df.csv", index_col=0)
df_long_a_enc = pd.read_csv(r"F:\Thesis\Data\Clean\df_long_a_enc_df.csv", index_col=0)

xgb_param = {'learning_rate' : [0.2], 
             'objective' : ['reg:squarederror'], 
             'n_estimators' : [200,500,1000],
             'reg_lambda' : [0,1,2],
             'subsample' : [0.3,0.6,0.9],
             'colsample_bytree' : [0.3,0.6,0.9],
             'max_depth' : [2,6,10]}

var_int = 'i_rate_growth'

d_p = df_long_p_enc.loc[:,var_int]
X_p_d = df_long_p_enc.iloc[:,4:]
X_p_d = X_p_d.drop(var_int, axis = 1)

xgb_tune_cv_d_p = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                           xgb_param, 
                           scoring = ['neg_mean_squared_error', 'r2'],  
                           cv=2, 
                           refit='r2', 
                           verbose = 1,
                           n_jobs=-1) 

a = time.time()
xgb_tune_cv_d_p.fit(X_p_d, d_p)
time_cv_d_p = time.time()-a

best_params_d_p = xgb_tune_cv_d_p.best_params_

best_score_d_p = xgb_tune_cv_d_p.best_score_

print('R2: ', r2_score(y_pred = xgb_tune_cv_d_p.predict(X_p_d), y_true = d_p))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_d_p.predict(X_p_d), y_true = d_p))

xgb_param_eta_d_p = {'learning_rate' : np.arange(0.01, 0.201, 0.01), 
                     'objective' : ['reg:squarederror'], 
                     'n_estimators' : [best_params_d_p['n_estimators']],
                     'reg_lambda' : [best_params_d_p['reg_lambda']],
                     'subsample' : [best_params_d_p['subsample']],
                     'colsample_bytree' : [best_params_d_p['colsample_bytree']],
                     'max_depth' : [best_params_d_p['max_depth']]}

xgb_tune_cv_eta_d_p = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                                   xgb_param_eta_d_p, 
                                   scoring=['neg_mean_squared_error', 'r2'],  
                                   cv=2, 
                                   refit='r2', 
                                   verbose = 1,
                                   n_jobs=-1)

a = time.time()
xgb_tune_cv_eta_d_p.fit(X_p_d, d_p)
time_cv_eta_d_p = time.time()-a

best_params_eta_d_p = xgb_tune_cv_eta_d_p.best_params_
best_score_eta_d_p = xgb_tune_cv_eta_d_p.best_score_

print('R2: ', r2_score(y_pred = xgb_tune_cv_eta_d_p.predict(X_p_d), y_true = d_p))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_eta_d_p.predict(X_p_d), y_true = d_p))

np.save("best_params_eta_d_p.npy", best_params_eta_d_p)

#%% D-task
### Active

d_a = df_long_a_enc.loc[:,var_int]
X_a_d = df_long_a_enc.iloc[:,4:]
X_a_d = X_a_d.drop(var_int, axis = 1)

xgb_tune_cv_d_a = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                           xgb_param, 
                           scoring=['neg_mean_squared_error', 'r2'],  
                           cv=2, 
                           refit='r2', 
                           verbose = 0,
                           n_jobs=-1) 

a = time.time()
xgb_tune_cv_d_a.fit(X_a_d, d_a)
time_cv_d_a = time.time()-a

best_params_d_a = xgb_tune_cv_d_a.best_params_
best_score_d_a = xgb_tune_cv_d_a.best_score_

print('R2: ', r2_score(y_pred = xgb_tune_cv_d_a.predict(X_a_d), y_true = d_a))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_d_a.predict(X_a_d), y_true = d_a))

xgb_param_eta_d_a = {'learning_rate' : np.arange(0.01, 0.301, 0.01), 
                     'objective' : ['reg:squarederror'], 
                     'n_estimators' : [best_params_d_a['n_estimators']],
                     'reg_lambda' : [best_params_d_a['reg_lambda']],
                     'subsample' : [best_params_d_a['subsample']],
                     'colsample_bytree' : [best_params_d_a['colsample_bytree']],
                     'max_depth' : [best_params_d_a['max_depth']]}

xgb_tune_cv_eta_d_a = GridSearchCV(XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor'), 
                                   xgb_param_eta_d_a, 
                                   scoring=['neg_mean_squared_error', 'r2'],  
                                   cv=2, 
                                   refit='r2', 
                                   verbose = 0,
                                   n_jobs=-1)

a = time.time()
xgb_tune_cv_eta_d_a.fit(X_a_d, d_a)
time_cv_eta_d_a = time.time()-a

best_params_eta_d_a = xgb_tune_cv_eta_d_a.best_params_
best_score_eta_d_a = xgb_tune_cv_eta_d_a.best_score_

print('R2: ', r2_score(y_pred = xgb_tune_cv_eta_d_a.predict(X_a_d), y_true = d_a))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_eta_d_a.predict(X_a_d), y_true = d_a))

# =============================================================================
#%% Splitting the data
# =============================================================================

df_long_p_i1 = df_long_p.sample(frac = 0.5)
df_long_p_i2 = df_long_p.drop(df_long_p_i1.index,axis=0)

df_long_a_i1 = df_long_a.sample(frac = 0.5)
df_long_a_i2 = df_long_a.drop(df_long_a_i1.index,axis=0)




# =============================================================================
#%% Normalizting the X_df
# =============================================================================

std_scaler = StandardScaler()
 
df_long_p_i2.iloc[:,4:12] = std_scaler.fit_transform(df_long_p_i2.iloc[:,4:12].to_numpy())
df_long_p_i1.iloc[:,4:12] = std_scaler.fit_transform(df_long_p_i1.iloc[:,4:12].to_numpy())

df_long_a_i2.iloc[:,4:12] = std_scaler.fit_transform(df_long_a_i2.iloc[:,4:12].to_numpy())
df_long_a_i1.iloc[:,4:12] = std_scaler.fit_transform(df_long_a_i1.iloc[:,4:12].to_numpy())

# =============================================================================
#%% Creating X and y for I1 and I2 for the dfs
# =============================================================================

X_p_i2 = df_long_p_i2.iloc[:,4:]
y_p_i2 = df_long_p_i2.iloc[:,1]

X_p_i1 = df_long_p_i1.iloc[:,4:]
y_p_i1 = df_long_p_i1.iloc[:,1]

X_a_i2 = df_long_a_i2.iloc[:,4:]
y_a_i2 = df_long_a_i2.iloc[:,1]

X_a_i1 = df_long_a_i1.iloc[:,4:]
y_a_i1 = df_long_a_i1.iloc[:,1]

# =============================================================================
#%% Linear regression (y)
# =============================================================================

# =============================================================================
# Fold 1:
# =============================================================================
model_p_y_1 = linear_model.LinearRegression()
model_p_y_1.fit(X_p_i2, y_p_i2)
print('Variance score: {}'.format(model_p_y_1.score(X_p_i1, y_p_i1)))


model_a_y_1 = linear_model.LinearRegression()
model_a_y_1.fit(X_a_i2, y_a_i2)
print('Variance score: {}'.format(model_a_y_1.score(X_a_i1, y_a_i1)))

# =============================================================================
# Fold 2:
# =============================================================================
model_p_y_2 = linear_model.LinearRegression()
model_p_y_2.fit(X_p_i1, y_p_i1)
print('Variance score: {}'.format(model_p_y_2.score(X_p_i2, y_p_i2)))


model_a_y_2 = linear_model.LinearRegression()
model_a_y_2.fit(X_a_i1, y_a_i1)
print('Variance score: {}'.format(model_a_y_2.score(X_a_i2, y_a_i2)))

# =============================================================================
#%% Creating the inputs for d-task (d)
# =============================================================================

var_int = 'i_rate_growth'

D_d_p_i2 = X_p_i2.loc[:,var_int]
D_X_p_i2 = X_p_i2
D_X_p_i2 = D_X_p_i2.drop(var_int, axis = 1)

D_d_p_i1 = X_p_i1.loc[:,var_int]
D_X_p_i1 = X_p_i1
D_X_p_i1 = D_X_p_i1.drop(var_int, axis = 1)

D_d_a_i2 = X_a_i2.loc[:,var_int]
D_X_a_i2 = X_a_i2
D_X_a_i2 = D_X_a_i2.drop(var_int, axis = 1)

D_d_a_i1 = X_a_i1.loc[:,var_int]
D_X_a_i1 = X_a_i1
D_X_a_i1 = D_X_a_i1.drop(var_int, axis = 1)

# =============================================================================
#%% Linear regression (d)
# =============================================================================

# =============================================================================
# Fold 1:
# =============================================================================
model_p_d_1 = linear_model.LinearRegression()
model_p_d_1.fit(D_X_p_i2, D_d_p_i2)
print('Variance score: {}'.format(model_p_d_1.score(D_X_p_i1, D_d_p_i1)))


model_a_d_1 = linear_model.LinearRegression()
model_a_d_1.fit(D_X_a_i2, D_d_a_i2)
print('Variance score: {}'.format(model_a_d_1.score(D_X_a_i1, D_d_a_i1)))

# =============================================================================
# Fold 2:
# =============================================================================

model_p_d_2 = linear_model.LinearRegression()
model_p_d_2.fit(D_X_p_i1, D_d_p_i1)
print('Variance score: {}'.format(model_p_d_2.score(D_X_p_i2, D_d_p_i2)))


model_a_d_2 = linear_model.LinearRegression()
model_a_d_2.fit(D_X_a_i1, D_d_a_i1)
print('Variance score: {}'.format(model_a_d_2.score(D_X_a_i2, D_d_a_i2)))

# =============================================================================
#%% Residuals
# =============================================================================

# =============================================================================
# Fold 1:
# =============================================================================
prediction_p_y_1 = model_p_y_1.predict(X_p_i1)
predictions_p_d_1 = model_p_d_1.predict(D_X_p_i1)

prediction_a_y_1 = model_a_y_1.predict(X_a_i1)
predictions_a_d_1 = model_a_d_1.predict(D_X_a_i1)


u_hat_p_1 = y_p_i1 - prediction_p_y_1
v_hat_p_1 = D_d_p_i1 - predictions_p_d_1

u_hat_a_1 = y_a_i1 - prediction_a_y_1
v_hat_a_1 = D_d_a_i1 - predictions_a_d_1

# =============================================================================
# Fold 2:
# =============================================================================
prediction_p_y_2 = model_p_y_2.predict(X_p_i2)
predictions_p_d_2 = model_p_d_2.predict(D_X_p_i2)

prediction_a_y_2 = model_a_y_2.predict(X_a_i2)
predictions_a_d_2 = model_a_d_2.predict(D_X_a_i2)


u_hat_p_2 = y_p_i2 - prediction_p_y_2
v_hat_p_2 = D_d_p_i2 - predictions_p_d_2

u_hat_a_2 = y_a_i2 - prediction_a_y_2
v_hat_a_2 = D_d_a_i2 - predictions_a_d_2


# =============================================================================
#%% Treatment effect
# =============================================================================

# =============================================================================
# Fold 1:
# =============================================================================
t_p_1 = sum(v_hat_p_1**2)**-1 * sum(u_hat_p_1 * v_hat_p_1)

t_a_1 = sum(v_hat_a_1**2)**-1 * sum(u_hat_a_1 * v_hat_a_1)


# =============================================================================
# Fold 2:
# =============================================================================
t_p_2 = sum(v_hat_p_2**2)**-1 * sum(u_hat_p_2 * v_hat_p_2)

t_a_2 = sum(v_hat_a_2**2)**-1 * sum(u_hat_a_2 * v_hat_a_2)


t_p_avg = (t_p_1 + t_p_2)/2
t_a_avg = (t_a_1 + t_a_2)/2








