# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 08:18:42 2022

@author: willi och gojja
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import time

df_long_p_enc = pd.read_csv(r"...\Data\Clean\df_long_p_enc_df.csv", index_col=0)
df_long_a_enc = pd.read_csv(r"...\Data\Clean\df_long_a_enc_df.csv", index_col=0)

# =============================================================================
#%% Tuning parameters for XGBoost 
# =============================================================================

#%% Y-task
### Passive

X_p = df_long_p_enc.iloc[:,4:]
y_p = df_long_p_enc.iloc[:,3]
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
                             pre_dispatch=4)#turning to +1 lets you see at what iteration it is currently, but using -1 allows the use of all cores. 

# os.environ['JOBLIB_TEMP_FOLDER'] = r'F:\Temp' #necessary for being able to handle the large intermediate results. IF this doesnt work try pathing it to F

a = time.time()
# with Parallel(max_nbytes=None): #Due to limitations in joblib::Parallels we need to use this fix so that the calculations do not consume all available disk memory. This was not necessary once I pathed the intermediate results to disk that had 500gb free. If you dont have it you should use the Parallel() trick.
xgb_tune_cv_y_p.fit(X_p, y_p)
time_cv_y_p = time.time()-a

best_params_y_p = xgb_tune_cv_y_p.best_params_

print('R2: ', r2_score(y_pred = xgb_tune_cv_y_p.predict(X_p), y_true = y_p))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_y_p.predict(X_p), y_true = y_p))

xgb_param_eta_y_p = {'learning_rate' : [best_params_y_p['learning_rate']], #np.arange(0.01, 0.201, 0.01)
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
                                   n_jobs=-1) 

a = time.time()
xgb_tune_cv_eta_y_p.fit(X_p, y_p)
time_cv_eta_y_p = time.time()-a

best_params_eta_y_p = xgb_tune_cv_eta_y_p.best_params_

print('R2: ', r2_score(y_pred = xgb_tune_cv_eta_y_p.predict(X_p), y_true = y_p))
print('MSE: ', mean_squared_error(y_pred = xgb_tune_cv_eta_y_p.predict(X_p), y_true = y_p))

np.save("best_params_eta_y_p.npy", best_params_eta_y_p)
best_params_y_p = np.load('best_params_eta_y_p.npy',allow_pickle='TRUE').item()

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
df_long_p_enc = pd.read_csv(r"...\Data\Clean\df_long_p_enc_df.csv", index_col=0)
df_long_a_enc = pd.read_csv(r"...\Data\Clean\df_long_a_enc_df.csv", index_col=0)

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
                           verbose = 1,
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