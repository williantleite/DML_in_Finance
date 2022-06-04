# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:22:40 2022

@author: gojja och willi

DISCLAIMER: In many occasions "old_data" is mentioned here. This alludes to a period of the development where we used a slightly different dataset. If your intention is to replicate the results presented in the thesis, we advice you to ignore the "old_data" lines completely and work only with those that reference "new_data". "old_data" was kept only for future research purposes.
"""

# =============================================================================
#%% Packages
# =============================================================================

import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.base import clone
from sklearn import linear_model
import numpy as np
from xgboost.sklearn import XGBRegressor
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import xgboost as xgb

# =============================================================================
#%% Data 
# =============================================================================

## Old data
# df_long_p = pd.read_csv(r".../Data/Fund data/df_long_p.csv", index_col=0)
# y_col_p = df_long_p.columns[0]
# d_cols_p = df_long_p.columns[10]
# x_cols_p = df_long_p.columns[3:].tolist()
# x_cols_p.remove(d_cols_p)
# df_long_p = df_long_p.drop(['year', 'month'], axis=1)

# dml_data_p = DoubleMLData(df_long_p,
#                           y_col = y_col_p,
#                           d_cols = d_cols_p,
#                           x_cols = x_cols_p)

# print(dml_data_p)

# df_long_a = pd.read_csv(r".../Data/Fund data/df_long_a.csv", index_col=0)
# y_col_a = df_long_a.columns[0]
# d_cols_a = df_long_a.columns[10]
# x_cols_a = df_long_a.columns[3:].tolist()
# x_cols_a.remove(d_cols_a)
# df_long_a = df_long_a.drop(['year', 'month'], axis=1)


# dml_data_a = DoubleMLData(df_long_a,
#                           y_col = y_col_a,
#                           d_cols = d_cols_a,
#                           x_cols = x_cols_a)

# print(dml_data_a)

# New data
df_long_new_p = pd.read_csv(r".../Data/Clean/df_long_p_enc_df.csv", index_col=0)
y_col_new_p = df_long_new_p.columns[3]
d_cols_new_p = df_long_new_p.columns[18]
x_cols_new_p = df_long_new_p.columns[4:].tolist()
x_cols_new_p.remove(d_cols_new_p)

df_long_new_p.index = df_long_new_p['variable']
df_long_new_p = df_long_new_p.drop(['year', 'month', 'variable'], axis=1)


dml_data_new_p = DoubleMLData(df_long_new_p,
                          y_col = y_col_new_p,
                          d_cols = d_cols_new_p,
                          x_cols = x_cols_new_p)

print(dml_data_new_p)

df_long_new_a = pd.read_csv(r".../Data/Clean/df_long_a_enc_df.csv", index_col=0)
y_col_new_a = df_long_new_a.columns[3]
d_cols_new_a = df_long_new_a.columns[18]
x_cols_new_a = df_long_new_a.columns[4:].tolist()
x_cols_new_a.remove(d_cols_new_a)

df_long_new_a.index = df_long_new_a['variable']
df_long_new_a = df_long_new_a.drop(['year', 'month', 'variable'], axis=1)


dml_data_new_a = DoubleMLData(df_long_new_a,
                          y_col = y_col_new_a,
                          d_cols = d_cols_new_a,
                          x_cols = x_cols_new_a)

print(dml_data_new_a)

best_params_eta_y_p = np.load('best_params_eta_y_p.npy',allow_pickle='TRUE').item()
best_params_eta_d_p = np.load('best_params_eta_d_p.npy',allow_pickle='TRUE').item()

best_params_eta_y_a = np.load('best_params_eta_y_a.npy',allow_pickle='TRUE').item()
best_params_eta_d_a = np.load('best_params_eta_d_a.npy',allow_pickle='TRUE').item()

dml_xgb_param_y_p = {'learning_rate' : best_params_eta_y_p['learning_rate'], 
                     'objective' : ['reg:squarederror'], 
                     'reg_lambda' : [best_params_eta_y_p['reg_lambda']],
                     'subsample' : [best_params_eta_y_p['subsample']],
                     'colsample_bytree' : [best_params_eta_y_p['colsample_bytree']],
                     'max_depth' : [best_params_eta_y_p['max_depth']],
                     'n_estimators' : [best_params_eta_y_p['n_estimators']]}

dml_xgb_param_d_p = {'learning_rate' : best_params_eta_d_p['learning_rate'], 
                     'objective' : ['reg:squarederror'], 
                     'reg_lambda' : [best_params_eta_d_p['reg_lambda']],
                     'subsample' : [best_params_eta_d_p['subsample']],
                     'colsample_bytree' : [best_params_eta_d_p['colsample_bytree']],
                     'max_depth' : [best_params_eta_d_p['max_depth']],
                     'n_estimators' : [best_params_eta_d_p['n_estimators']]}

dml_xgb_param_y_a = {'learning_rate' : best_params_eta_y_a['learning_rate'], 
                     'objective' : ['reg:squarederror'], 
                     'reg_lambda' : [best_params_eta_y_a['reg_lambda']],
                     'subsample' : [best_params_eta_y_a['subsample']],
                     'colsample_bytree' : [best_params_eta_y_a['colsample_bytree']],
                     'max_depth' : [best_params_eta_y_a['max_depth']],
                     'n_estimators' : [best_params_eta_y_a['n_estimators']]}

dml_xgb_param_d_a = {'learning_rate' : best_params_eta_d_a['learning_rate'], 
                     'objective' : ['reg:squarederror'], 
                     'reg_lambda' : [best_params_eta_d_a['reg_lambda']],
                     'subsample' : [best_params_eta_d_a['subsample']],
                     'colsample_bytree' : [best_params_eta_d_a['colsample_bytree']],
                     'max_depth' : [best_params_eta_d_a['max_depth']],
                     'n_estimators' : [best_params_eta_d_a['n_estimators']]}

# =============================================================================
#%% DML (passive)
# =============================================================================
learner_y_p = XGBRegressor(eta = dml_xgb_param_y_p['learning_rate'],
                           objective = dml_xgb_param_y_p['objective'][0],
                           reg_lambda = dml_xgb_param_y_p['reg_lambda'][0],
                           subsample = dml_xgb_param_y_p['subsample'][0],
                           colsample_bytree = dml_xgb_param_y_p['colsample_bytree'][0],
                           max_depth = dml_xgb_param_y_p['max_depth'][0],
                           n_estimators = dml_xgb_param_y_p['n_estimators'][0])

learner_d_p = XGBRegressor(eta = dml_xgb_param_d_p['learning_rate'],
                           objective = dml_xgb_param_y_p['objective'][0],
                           reg_lambda = dml_xgb_param_d_p['reg_lambda'][0],
                           subsample = dml_xgb_param_d_p['subsample'][0],
                           colsample_bytree = dml_xgb_param_d_p['colsample_bytree'][0],
                           max_depth = dml_xgb_param_d_p['max_depth'][0],
                           n_estimators = dml_xgb_param_d_p['n_estimators'][0])
ml_g_p = clone(learner_y_p)
ml_m_p = clone(learner_d_p)

# obj_dml_plr_p = DoubleMLPLR(dml_data_p, ml_g_p, ml_m_p)
# obj_dml_plr_p.fit()

# print(obj_dml_plr_p)

# =============================================================================
#%% DML (active)
# =============================================================================
learner_y_a = XGBRegressor(eta = dml_xgb_param_y_a['learning_rate'],
                           objective = dml_xgb_param_y_p['objective'][0],
                           reg_lambda = dml_xgb_param_y_a['reg_lambda'][0],
                           subsample = dml_xgb_param_y_a['subsample'][0],
                           colsample_bytree = dml_xgb_param_y_a['colsample_bytree'][0],
                           max_depth = dml_xgb_param_y_a['max_depth'][0],
                           n_estimators = dml_xgb_param_y_a['n_estimators'][0])

learner_d_a = XGBRegressor(eta = dml_xgb_param_d_a['learning_rate'],
                           objective = dml_xgb_param_y_p['objective'][0],
                           reg_lambda = dml_xgb_param_d_a['reg_lambda'][0],
                           subsample = dml_xgb_param_d_a['subsample'][0],
                           colsample_bytree = dml_xgb_param_d_a['colsample_bytree'][0],
                           max_depth = dml_xgb_param_d_a['max_depth'][0],
                           n_estimators = dml_xgb_param_d_a['n_estimators'][0])
ml_g_a = clone(learner_y_a)
ml_m_a = clone(learner_d_a)

# obj_dml_plr_a = DoubleMLPLR(dml_data_a, ml_g_a, ml_m_a)
# obj_dml_plr_a.fit()
# obj_dml_plr_a = obj_dml_plr_a

# print(obj_dml_plr_a)


# =============================================================================
#%% DML function:
# =============================================================================

def dml_any(df, learner_y, learner_d):
    start = time.time()
    dml_obj = DoubleMLPLR(df, learner_y, learner_d)
    dml_fin = dml_obj.fit(store_predictions=True)
    end = time.time()
    print('Time of function:' + str(end-start))
    return(dml_fin)

# =============================================================================
#%% Linear modeling with DML package
# =============================================================================

model_y = linear_model.LinearRegression()
model_d = linear_model.LinearRegression()

# lin_p = dml_any(dml_data_p,model_y,model_d)
# lin_a = dml_any(dml_data_a,model_y,model_d)

# print(lin_p)
# print(lin_a)

#Here the results are repeated 10 times as a means to compare how consistent the results are:

lin_new_p_1 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_2 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_3 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_4 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_5 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_6 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_7 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_8 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_9 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_10 = dml_any(dml_data_new_p,model_y,model_d)

print(lin_new_p_1)
print(lin_new_p_2)
print(lin_new_p_3)
print(lin_new_p_4)
print(lin_new_p_5)
print(lin_new_p_6)
print(lin_new_p_7)
print(lin_new_p_8)
print(lin_new_p_9)
print(lin_new_p_10)

avg_coef_lin_p = sum(lin_new_p_1.coef+lin_new_p_2.coef+lin_new_p_3.coef+
                     lin_new_p_4.coef+lin_new_p_5.coef+lin_new_p_6.coef+
                     lin_new_p_7.coef+lin_new_p_8.coef+lin_new_p_9.coef+
                     lin_new_p_10.coef)/10

# std_coef_lin_p = np.std(lin_new_p_1.coef+lin_new_p_2.coef+lin_new_p_3.coef+
#                      lin_new_p_4.coef+lin_new_p_5.coef+lin_new_p_6.coef+
#                      lin_new_p_7.coef+lin_new_p_8.coef+lin_new_p_9.coef+
#                      lin_new_p_10.coef)

avg_pval_lin_p = sum(lin_new_p_1.pval+lin_new_p_2.pval+lin_new_p_3.pval+
                     lin_new_p_4.pval+lin_new_p_5.pval+lin_new_p_6.pval+
                     lin_new_p_7.pval+lin_new_p_8.pval+lin_new_p_9.pval+
                     lin_new_p_10.pval)/10

# std_pval_lin_p = np.std(lin_new_p_1.pval+lin_new_p_2.pval+lin_new_p_3.pval+
#                      lin_new_p_4.pval+lin_new_p_5.pval+lin_new_p_6.pval+
#                      lin_new_p_7.pval+lin_new_p_8.pval+lin_new_p_9.pval+
#                      lin_new_p_10.pval)

# lin_new_a = dml_any(dml_data_new_a,model_y,model_d)

lin_new_a_1 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_2 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_3 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_4 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_5 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_6 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_7 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_8 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_9 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_10 = dml_any(dml_data_new_a,model_y,model_d)

print(lin_new_a_1)
print(lin_new_a_2)
print(lin_new_a_3)
print(lin_new_a_4)
print(lin_new_a_5)
print(lin_new_a_6)
print(lin_new_a_7)
print(lin_new_a_8)
print(lin_new_a_9)
print(lin_new_a_10)

avg_coef_lin_a = sum(lin_new_a_1.coef+lin_new_a_2.coef+lin_new_a_3.coef+
                     lin_new_a_4.coef+lin_new_a_5.coef+lin_new_a_6.coef+
                     lin_new_a_7.coef+lin_new_a_8.coef+lin_new_a_9.coef+
                     lin_new_a_10.coef)/10

# std_coef_lin_a = np.std(lin_new_a_1.coef+lin_new_a_2.coef+lin_new_a_3.coef+
#                      lin_new_a_4.coef+lin_new_a_5.coef+lin_new_a_6.coef+
#                      lin_new_a_7.coef+lin_new_a_8.coef+lin_new_a_9.coef+
#                      lin_new_a_10.coef)

avg_pval_lin_a = sum(lin_new_a_1.pval+lin_new_a_2.pval+lin_new_a_3.pval+
                     lin_new_a_4.pval+lin_new_a_5.pval+lin_new_a_6.pval+
                     lin_new_a_7.pval+lin_new_a_8.pval+lin_new_a_9.pval+
                     lin_new_a_10.pval)/10

# std_pval_lin_a = np.std(lin_new_a_1.pval+lin_new_a_2.pval+lin_new_a_3.pval+
#                      lin_new_a_4.pval+lin_new_a_5.pval+lin_new_a_6.pval+
#                      lin_new_a_7.pval+lin_new_a_8.pval+lin_new_a_9.pval+
#                      lin_new_a_10.pval)

# print(lin_new_p)
# print(lin_new_a)

# lin_new_p

lin_new_p_final_1 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_final_2 = dml_any(dml_data_new_p,model_y,model_d)
lin_new_p_final_3 = dml_any(dml_data_new_p,model_y,model_d)

lin_new_a_final_1 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_final_2 = dml_any(dml_data_new_a,model_y,model_d)
lin_new_a_final_3 = dml_any(dml_data_new_a,model_y,model_d)

# Getting the residuals vs i_rate  and the residuals vs fitted for the LR

pre_lin_p_1 = lin_new_p_final_1.predictions
pre_lin_p_2 = lin_new_p_final_2.predictions
pre_lin_p_3 = lin_new_p_final_3.predictions

pre_lin_a_1 = lin_new_a_final_1.predictions
pre_lin_a_2 = lin_new_a_final_2.predictions
pre_lin_a_3 = lin_new_a_final_3.predictions

pre_lin_p_ml_g_1 = pre_lin_p_1["ml_g"][:,0,0]
pre_lin_p_ml_m_1 = pre_lin_p_1["ml_m"][:,0,0]

pre_lin_p_ml_g_2 = pre_lin_p_2["ml_g"][:,0,0]
pre_lin_p_ml_m_2 = pre_lin_p_2["ml_m"][:,0,0]

pre_lin_p_ml_g_3 = pre_lin_p_3["ml_g"][:,0,0]
pre_lin_p_ml_m_3 = pre_lin_p_3["ml_m"][:,0,0]

pre_lin_a_ml_g_1 = pre_lin_a_1["ml_g"][:,0,0]
pre_lin_a_ml_m_1 = pre_lin_a_1["ml_m"][:,0,0]

pre_lin_a_ml_g_2 = pre_lin_a_2["ml_g"][:,0,0]
pre_lin_a_ml_m_2 = pre_lin_a_2["ml_m"][:,0,0]

pre_lin_a_ml_g_3 = pre_lin_a_3["ml_g"][:,0,0]
pre_lin_a_ml_m_3 = pre_lin_a_3["ml_m"][:,0,0]

i_rate_new_p = pd.read_csv(r".../Data/Clean/df_long_p_enc_df.csv", index_col=0)
i_rate_new_p  = pd.DataFrame(i_rate_new_p.iloc[:,18])

i_rate_new_a = pd.read_csv(r".../Data/Clean/df_long_a_enc_df.csv", index_col=0)
i_rate_new_a  = pd.DataFrame(i_rate_new_a.iloc[:,18])

res_lin_p_1 = pd.DataFrame(Y_p-pre_lin_p_ml_g_1)
res_lin_p_2 = pd.DataFrame(Y_p-pre_lin_p_ml_g_2)
res_lin_p_3 = pd.DataFrame(Y_p-pre_lin_p_ml_g_3)

res_lin_a_1 = pd.DataFrame(Y_a-pre_lin_a_ml_g_1)
res_lin_a_2 = pd.DataFrame(Y_a-pre_lin_a_ml_g_2)
res_lin_a_3 = pd.DataFrame(Y_a-pre_lin_a_ml_g_3)

df_lin_plot_p_1 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))
df_lin_plot_p_2 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))
df_lin_plot_p_3 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))

df_lin_plot_a_1 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))
df_lin_plot_a_2 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))
df_lin_plot_a_3 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))

df_lin_plot_p_1.iloc[:,0], df_lin_plot_p_1.iloc[:,1], df_lin_plot_p_1.iloc[:,2] = res_lin_p_1, i_rate_new_p, pre_lin_p_ml_g_1
df_lin_plot_p_2.iloc[:,0], df_lin_plot_p_2.iloc[:,1], df_lin_plot_p_2.iloc[:,2] = res_lin_p_2, i_rate_new_p, pre_lin_p_ml_g_2
df_lin_plot_p_3.iloc[:,0], df_lin_plot_p_3.iloc[:,1], df_lin_plot_p_3.iloc[:,2] = res_lin_p_3, i_rate_new_p, pre_lin_p_ml_g_3

df_lin_plot_a_1.iloc[:,0], df_lin_plot_a_1.iloc[:,1], df_lin_plot_a_1.iloc[:,2] = res_lin_a_1, i_rate_new_a, pre_lin_a_ml_g_1
df_lin_plot_a_2.iloc[:,0], df_lin_plot_a_2.iloc[:,1], df_lin_plot_a_2.iloc[:,2] = res_lin_a_2, i_rate_new_a, pre_lin_a_ml_g_2
df_lin_plot_a_3.iloc[:,0], df_lin_plot_a_3.iloc[:,1], df_lin_plot_a_3.iloc[:,2] = res_lin_a_3, i_rate_new_a, pre_lin_a_ml_g_3

df_lin_plot_p_1.columns = ['Residuals','i_rate_growth','Fitted values']
df_lin_plot_p_2.columns = ['Residuals','i_rate_growth','Fitted values']
df_lin_plot_p_3.columns = ['Residuals','i_rate_growth','Fitted values']

df_lin_plot_a_1.columns = ['Residuals','i_rate_growth','Fitted values']
df_lin_plot_a_2.columns = ['Residuals','i_rate_growth','Fitted values']
df_lin_plot_a_3.columns = ['Residuals','i_rate_growth','Fitted values']

# Plotting residuals vs i_rate for the LR
df_lin_plot_p_1.plot(x='i_rate_growth', y='Residuals', style='o')
df_lin_plot_p_2.plot(x='i_rate_growth', y='Residuals', style='o')
df_lin_plot_p_3.plot(x='i_rate_growth', y='Residuals', style='o')

df_lin_plot_a_1.plot(x='i_rate_growth', y='Residuals', style='o')
df_lin_plot_a_2.plot(x='i_rate_growth', y='Residuals', style='o')
df_lin_plot_a_3.plot(x='i_rate_growth', y='Residuals', style='o')

# Getting the residuals vs fitted for the LR
df_lin_plot_p_1.plot(x='Fitted values', y='Residuals', style='o')
df_lin_plot_p_2.plot(x='Fitted values', y='Residuals', style='o')
df_lin_plot_p_3.plot(x='Fitted values', y='Residuals', style='o')

df_lin_plot_a_1.plot(x='Fitted values', y='Residuals', style='o')
df_lin_plot_a_2.plot(x='Fitted values', y='Residuals', style='o')
df_lin_plot_a_3.plot(x='Fitted values', y='Residuals', style='o')

# =============================================================================
#%% XGB with DML package (old data)
# =============================================================================

# xgb_old_data_p = dml_any(dml_data_p,ml_g_a,ml_m_a)
# print(xgb_old_data_p)

# xgb_old_data_a = dml_any(dml_data_a,ml_g_a,ml_m_a)
# print(xgb_old_data_a)

# =============================================================================
#%% XGB with DML package (new data)
# =============================================================================

xgb_new_data_p_1 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_p_2 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_p_3 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_p_4 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_p_5 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
# xgb_new_data_p_6 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
# xgb_new_data_p_7 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
# xgb_new_data_p_8 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
# xgb_new_data_p_9 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
# xgb_new_data_p_10 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)

xgb_new_data_p_final = dml_any(dml_data_new_p, ml_g_p, ml_m_p)

print(xgb_new_data_p_1.coef)
print(xgb_new_data_p_2.coef)
print(xgb_new_data_p_3.coef)
print(xgb_new_data_p_4.coef)
print(xgb_new_data_p_5.coef)

print(xgb_new_data_p_1)
print(xgb_new_data_p_2)
print(xgb_new_data_p_3)
print(xgb_new_data_p_4)
print(xgb_new_data_p_5)

print(xgb_new_data_p_1.predictions)

xgb_new_data_p_1.learner_names

avg_coef_xbg_p = sum(xgb_new_data_p_1.coef+xgb_new_data_p_2.coef+
                     xgb_new_data_p_3.coef+xgb_new_data_p_4.coef+
                     xgb_new_data_p_5.coef)/5

std_coef_xbg_p = np.std((xgb_new_data_p_1.coef,xgb_new_data_p_2.coef,
                     xgb_new_data_p_3.coef,xgb_new_data_p_4.coef,
                     xgb_new_data_p_5.coef))

avg_pval_xbg_p = sum(xgb_new_data_p_1.pval+xgb_new_data_p_2.pval+
                     xgb_new_data_p_3.pval+xgb_new_data_p_4.pval+
                     xgb_new_data_p_5.pval)/5

std_pval_xbg_p = np.std((xgb_new_data_p_1.pval,xgb_new_data_p_2.pval,
                     xgb_new_data_p_3.pval,xgb_new_data_p_4.pval,
                     xgb_new_data_p_5.pval))

# print(xgb_new_data_p)

xgb_new_data_a_1 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
xgb_new_data_a_2 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
xgb_new_data_a_3 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
xgb_new_data_a_4 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
xgb_new_data_a_5 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
# xgb_new_data_a_6 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
# xgb_new_data_a_7 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
# xgb_new_data_a_8 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
# xgb_new_data_a_9 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
# xgb_new_data_a_10 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)

print(xgb_new_data_a_1)
print(xgb_new_data_a_2)
print(xgb_new_data_a_3)
print(xgb_new_data_a_4)
print(xgb_new_data_a_5)

avg_coef_xbg_a = sum(xgb_new_data_a_1.coef+xgb_new_data_a_2.coef+
                     xgb_new_data_a_3.coef+xgb_new_data_a_4.coef+
                     xgb_new_data_a_5.coef)/5

std_coef_xbg_a = np.std((xgb_new_data_a_1.coef,xgb_new_data_a_2.coef,
                     xgb_new_data_a_3.coef,xgb_new_data_a_4.coef,
                     xgb_new_data_a_5.coef))

avg_pval_xbg_a = sum(xgb_new_data_a_1.pval+xgb_new_data_a_2.pval+
                     xgb_new_data_a_3.pval+xgb_new_data_a_4.pval+
                     xgb_new_data_a_5.pval)/5

std_pval_xbg_a = np.std((xgb_new_data_a_1.pval,xgb_new_data_a_2.pval,
                     xgb_new_data_a_3.pval,xgb_new_data_a_4.pval,
                     xgb_new_data_a_5.pval))

# print(xgb_new_data_a)

print(xgb_new_data_p_1.predictions)

# Runing it a final time but with updataed DML to save predictions:
xgb_new_data_p_final_1 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_a_final_1 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
xgb_new_data_p_final_2 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_a_final_2 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)
xgb_new_data_p_final_3 = dml_any(dml_data_new_p, ml_g_p, ml_m_p)
xgb_new_data_a_final_3 = dml_any(dml_data_new_a,ml_g_a,ml_m_a)

print(xgb_new_data_p_final_1)
print(xgb_new_data_p_final_2)
print(xgb_new_data_p_final_3)

print(xgb_new_data_a_final_1)
print(xgb_new_data_a_final_2)
print(xgb_new_data_a_final_3)

pre_p_1 = xgb_new_data_p_final_1.predictions
pre_p_2 = xgb_new_data_p_final_2.predictions
pre_p_3 = xgb_new_data_p_final_3.predictions

pre_a_1 = xgb_new_data_a_final_1.predictions
pre_a_2 = xgb_new_data_a_final_2.predictions
pre_a_3 = xgb_new_data_a_final_3.predictions

pre_p_ml_g_1 = pre_p_1["ml_g"][:,0,0]
pre_p_ml_m_1 = pre_p_1["ml_m"][:,0,0]

pre_p_ml_g_2 = pre_p_2["ml_g"][:,0,0]
pre_p_ml_m_2 = pre_p_2["ml_m"][:,0,0]

pre_p_ml_g_3 = pre_p_3["ml_g"][:,0,0]
pre_p_ml_m_3 = pre_p_3["ml_m"][:,0,0]

pre_a_ml_g_1 = pre_a_1["ml_g"][:,0,0]
pre_a_ml_m_1 = pre_a_1["ml_m"][:,0,0]

pre_a_ml_g_2 = pre_a_2["ml_g"][:,0,0]
pre_a_ml_m_2 = pre_a_2["ml_m"][:,0,0]

pre_a_ml_g_3 = pre_a_3["ml_g"][:,0,0]
pre_a_ml_m_3 = pre_a_3["ml_m"][:,0,0]

# Getting the residuals vs i_rate for the xgb
i_rate_new_p = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/full_datasets/df_long_p_enc_df.csv", index_col=0)
i_rate_new_p  = pd.DataFrame(i_rate_new_p.iloc[:,18])

i_rate_new_a = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/full_datasets/df_long_a_enc_df.csv", index_col=0)
i_rate_new_a  = pd.DataFrame(i_rate_new_a.iloc[:,18])

res_p_1 = pd.DataFrame(Y_p-pre_p_ml_g_1)
res_p_2 = pd.DataFrame(Y_p-pre_p_ml_g_2)
res_p_3 = pd.DataFrame(Y_p-pre_p_ml_g_3)

res_a_1 = pd.DataFrame(Y_a-pre_a_ml_g_1)
res_a_2 = pd.DataFrame(Y_a-pre_a_ml_g_2)
res_a_3 = pd.DataFrame(Y_a-pre_a_ml_g_3)

df_plot_p_1 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))
df_plot_p_2 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))
df_plot_p_3 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))

df_plot_a_1 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))
df_plot_a_2 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))
df_plot_a_3 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))

df_plot_p_1.iloc[:,0], df_plot_p_1.iloc[:,1], df_plot_p_1.iloc[:,2] = res_p_1, i_rate_new_p, pre_p_ml_g_1
df_plot_p_2.iloc[:,0], df_plot_p_2.iloc[:,1], df_plot_p_2.iloc[:,2] = res_p_2, i_rate_new_p, pre_p_ml_g_2
df_plot_p_3.iloc[:,0], df_plot_p_3.iloc[:,1], df_plot_p_3.iloc[:,2] = res_p_3, i_rate_new_p, pre_p_ml_g_3

df_plot_a_1.iloc[:,0], df_plot_a_1.iloc[:,1], df_plot_a_1.iloc[:,2] = res_a_1, i_rate_new_a, pre_a_ml_g_1
df_plot_a_2.iloc[:,0], df_plot_a_2.iloc[:,1], df_plot_a_2.iloc[:,2] = res_a_2, i_rate_new_a, pre_a_ml_g_2
df_plot_a_3.iloc[:,0], df_plot_a_3.iloc[:,1], df_plot_a_3.iloc[:,2] = res_a_3, i_rate_new_a, pre_a_ml_g_3

df_plot_p_1.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_p_2.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_p_3.columns = ['Residuals','i_rate_growth','Fitted values']

df_plot_a_1.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_a_2.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_a_3.columns = ['Residuals','i_rate_growth','Fitted values']

df_plot_p_1.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_p_2.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_p_3.plot(x='i_rate_growth', y='Residuals', style='o')

df_plot_a_1.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_a_2.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_a_3.plot(x='i_rate_growth', y='Residuals', style='o', ylim=(-1,1))

# Getting the residuals vs fitted for the xgb
df_plot_p_1.plot(x='Fitted values', y='Residuals', style='o')
df_plot_p_2.plot(x='Fitted values', y='Residuals', style='o')
df_plot_p_3.plot(x='Fitted values', y='Residuals', style='o')

df_plot_a_1.plot(x='Fitted values', y='Residuals', style='o')
df_plot_a_2.plot(x='Fitted values', y='Residuals', style='o')
df_plot_a_3.plot(x='Fitted values', y='Residuals', style='o')

# For the cross m function with D=Y
res_p_1 = pd.DataFrame(D_p-pre_p_ml_m_1)
res_p_2 = pd.DataFrame(D_p-pre_p_ml_m_2)
res_p_3 = pd.DataFrame(D_p-pre_p_ml_m_3)

res_a_1 = pd.DataFrame(D_a-pre_a_ml_m_1)
res_a_2 = pd.DataFrame(D_a-pre_a_ml_m_2)
res_a_3 = pd.DataFrame(D_a-pre_a_ml_m_3)

df_plot_p_1 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))
df_plot_p_2 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))
df_plot_p_3 = pd.DataFrame(np.zeros((len(i_rate_new_p),3)))

df_plot_a_1 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))
df_plot_a_2 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))
df_plot_a_3 = pd.DataFrame(np.zeros((len(i_rate_new_a),3)))

df_plot_p_1.iloc[:,0], df_plot_p_1.iloc[:,1], df_plot_p_1.iloc[:,2] = res_p_1, i_rate_new_p, pre_p_ml_g_1
df_plot_p_2.iloc[:,0], df_plot_p_2.iloc[:,1], df_plot_p_2.iloc[:,2] = res_p_2, i_rate_new_p, pre_p_ml_g_2
df_plot_p_3.iloc[:,0], df_plot_p_3.iloc[:,1], df_plot_p_3.iloc[:,2] = res_p_3, i_rate_new_p, pre_p_ml_g_3

df_plot_a_1.iloc[:,0], df_plot_a_1.iloc[:,1], df_plot_a_1.iloc[:,2] = res_a_1, i_rate_new_a, pre_a_ml_g_1
df_plot_a_2.iloc[:,0], df_plot_a_2.iloc[:,1], df_plot_a_2.iloc[:,2] = res_a_2, i_rate_new_a, pre_a_ml_g_2
df_plot_a_3.iloc[:,0], df_plot_a_3.iloc[:,1], df_plot_a_3.iloc[:,2] = res_a_3, i_rate_new_a, pre_a_ml_g_3

df_plot_p_1.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_p_2.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_p_3.columns = ['Residuals','i_rate_growth','Fitted values']

df_plot_a_1.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_a_2.columns = ['Residuals','i_rate_growth','Fitted values']
df_plot_a_3.columns = ['Residuals','i_rate_growth','Fitted values']

df_plot_p_1.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_p_2.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_p_3.plot(x='i_rate_growth', y='Residuals', style='o')

df_plot_a_1.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_a_2.plot(x='i_rate_growth', y='Residuals', style='o')
df_plot_a_3.plot(x='i_rate_growth', y='Residuals', style='o')

# Getting the residuals vs fitted for the xgb
df_plot_p_1.plot(x='Fitted values', y='Residuals', style='o')
df_plot_p_2.plot(x='Fitted values', y='Residuals', style='o')
df_plot_p_3.plot(x='Fitted values', y='Residuals', style='o')

df_plot_a_1.plot(x='Fitted values', y='Residuals', style='o')
df_plot_a_2.plot(x='Fitted values', y='Residuals', style='o')
df_plot_a_3.plot(x='Fitted values', y='Residuals', style='o')

# =============================================================================
#%% Runing a LR on the data
# =============================================================================

X_p = df_long_new_p.iloc[:,1:]
Y_p = df_long_new_p.iloc[:,0]

X_a = df_long_new_a.iloc[:,1:]
Y_a = df_long_new_a.iloc[:,0]

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, Y_p, test_size=0.3,
                                                            random_state=1)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, Y_a, test_size=0.3,
                                                            random_state=1)

# Y-task 

model_lr_test_y_p = linear_model.LinearRegression()
model_lr_test_y_a = linear_model.LinearRegression()

model_lr_test_y_p.fit(X_train_p, y_train_p)
model_lr_test_y_a.fit(X_train_a, y_train_a)

# regression coefficients
print('Coefficients (passive): ', model_lr_test_y_p.coef_)
print('Coefficients (active): ', model_lr_test_y_a.coef_)

# variance score: 1 means perfect prediction

score_lr_y_p = model_lr_test_y_p.score(X_test_p, y_test_p)
score_lr_y_a = model_lr_test_y_a.score(X_test_a, y_test_a)

print('Variance score: ', score_lr_y_p)
print('Variance score: ', score_lr_y_a)

# D-task

X_train_d_p, X_test_d_p, D_train_d_p, D_test_d_p = train_test_split(X_p_d, D_p, test_size=0.3, random_state=1)

X_train_d_a, X_test_d_a, D_train_d_a, D_test_d_a = train_test_split(X_a_d, D_a, test_size=0.3, random_state=1)

model_lr_test_d_p = linear_model.LinearRegression()
model_lr_test_d_a = linear_model.LinearRegression()

model_lr_test_d_p.fit(X_train_d_p, D_train_d_p)
model_lr_test_d_a.fit(X_train_d_a, D_train_d_a)

score_lr_d_p = model_lr_test_d_p.score(X_test_d_p, D_test_d_p)
score_lr_d_a = model_lr_test_d_a.score(X_test_d_a, D_test_d_a)

print('Variance score: ', score_lr_d_p)
print('Variance score: ', score_lr_d_a)

# Doing the adjusted R2

score_lr_y_p_adj = 1-(1-score_lr_y_p)*((Y_p.shape[0]-1)/(Y_p.shape[0]-X_p.shape[1]-1))
score_lr_y_a_adj = 1-(1-score_lr_y_a)*((Y_a.shape[0]-1)/(Y_a.shape[0]-X_a.shape[1]-1))
score_lr_d_p_adj = 1-(1-score_lr_d_p)*((D_p.shape[0]-1)/(D_p.shape[0]-X_p_d.shape[1]-1))
score_lr_d_a_adj = 1-(1-score_lr_d_a)*((D_a.shape[0]-1)/(D_a.shape[0]-X_a_d.shape[1]-1))

print('Adjusted (y-passive): ', score_lr_y_p_adj)
print('Adjusted (y-active): ', score_lr_y_a_adj)
print('Adjusted (d-passive): ', score_lr_d_p_adj)
print('Adjusted (d-active: ', score_lr_d_a_adj)

# =============================================================================
#%% Runing a XGB on the data
# =============================================================================

ml_g_p = clone(learner_y_p)
ml_m_p = clone(learner_d_p)

ml_g_a = clone(learner_y_a)
ml_m_a = clone(learner_d_a)

df_long_new_p
df_long_new_a

# Data creation

X_p = df_long_new_p.iloc[:,1:]
Y_p = df_long_new_p.iloc[:,0]

X_a = df_long_new_a.iloc[:,1:]
Y_a = df_long_new_a.iloc[:,0]

X_p_d = X_p
X_p_d = X_p_d.drop(d_cols_new_p,axis = 1)
D_p = df_long_new_p.iloc[:,15]

X_a_d = X_a
X_a_d = X_a_d.drop(d_cols_new_a,axis = 1)
D_a = df_long_new_a.iloc[:,15]

X_train_y_p, X_test_y_p, y_train_y_p, y_test_y_p = train_test_split(X_p, Y_p, 
                                                                    test_size=0.3,
                                                                    random_state=1)

X_train_y_a, X_test_y_a, y_train_y_a, y_test_y_a = train_test_split(X_a, Y_a, 
                                                                    test_size=0.3, 
                                                                    random_state=1)

X_train_d_p, X_test_d_p, D_train_d_p, D_test_d_p = train_test_split(X_p_d, D_p, 
                                                                    test_size=0.3,
                                                                    random_state=1)

X_train_d_a, X_test_d_a, D_train_d_a, D_test_d_a = train_test_split(X_a_d, D_a, 
                                                                    test_size=0.3, 
                                                                    random_state=1)

# y-models

xgb_y_p = clone(ml_g_p)
xgb_y_a = clone(ml_g_a)

# d-models

xgb_d_p = clone(ml_m_p)
xgb_d_a = clone(ml_m_a)

# fitting the models

xgb_y_p.fit(X_train_y_p, y_train_y_p)
xgb_d_p.fit(X_train_d_p, D_train_d_p)

xgb_y_a.fit(X_train_y_a, y_train_y_a)
xgb_d_a.fit(X_train_d_a, D_train_d_a)

# R2 of the models

score_y_p = r2_score(y_pred = xgb_y_p.predict(X_p),y_true = Y_p)
score_y_a = r2_score(y_pred = xgb_y_a.predict(X_a),y_true = Y_a)
score_d_p = r2_score(y_pred = xgb_d_p.predict(X_p_d),y_true = D_p)
score_d_a = r2_score(y_pred = xgb_d_a.predict(X_a_d),y_true = D_a)

print("Training score (y-passive): ", score_y_p)
print("Training score (y-active): ", score_y_a)
print("Training score (d-passive): ", score_d_p)
print("Training score (d-active): ", score_d_a)

# Adjusted R2:
    
score_y_p_adj = 1-(1-score_y_p)*((Y_p.shape[0]-1)/(Y_p.shape[0]-X_p.shape[1]-1))
score_y_a_adj = 1-(1-score_y_a)*((Y_a.shape[0]-1)/(Y_a.shape[0]-X_a.shape[1]-1))
score_d_p_adj = 1-(1-score_d_p)*((D_p.shape[0]-1)/(D_p.shape[0]-X_p_d.shape[1]-1))
score_d_a_adj = 1-(1-score_d_a)*((D_a.shape[0]-1)/(D_a.shape[0]-X_a_d.shape[1]-1))

print("Adjusted (y-passive): ", score_y_p_adj)
print("Adjusted score (y-active): ", score_y_a_adj)
print("Adjusted score (d-passive): ", score_d_p_adj)
print("Adjusted score (d-active): ", score_d_a_adj)

# MSE
pred_xgb_y_p = xgb_y_p.predict(X_p)
pred_xgb_y_a = xgb_y_a.predict(X_a)

mse_xgb_y_p = sum((Y_p-pred_xgb_y_p)**2)/len(Y_p)
mse_xgb_y_a = sum((Y_a-pred_xgb_y_a)**2)/len(Y_a)

y_true = Y_p

# =============================================================================
#%% Variable imporance XGB
# =============================================================================

var_importances_y_p = pd.DataFrame(data={
    'Attribute': X_p.columns,
    'Importance': xgb_y_p.feature_importances_
})

var_importances_y_a = pd.DataFrame(data={
    'Attribute': X_a.columns,
    'Importance': xgb_y_a.feature_importances_
})

var_importances_d_p = pd.DataFrame(data={
    'Attribute': X_p_d.columns,
    'Importance': xgb_d_p.feature_importances_
})

var_importances_d_a = pd.DataFrame(data={
    'Attribute': X_a_d.columns,
    'Importance': xgb_d_a.feature_importances_
})

importances_y_p = var_importances_y_p.sort_values(by='Importance', ascending=False)
importances_y_a = var_importances_y_a.sort_values(by='Importance', ascending=False)

importances_d_p = var_importances_d_p.sort_values(by='Importance', ascending=False)
importances_d_a = var_importances_d_a.sort_values(by='Importance', ascending=False)

# Creating plots
# y_p
plt.bar(x=importances_y_p.iloc[:,0], height=importances_y_p.iloc[:,1], color='#087E8B')
plt.title('Feature importances (y-task, passive funds)', size=20)
plt.xticks(rotation='vertical')
plt.show()

plt.bar(x=importances_y_p.iloc[64:,0], height=importances_y_p.iloc[64:,1], color='#087E8B')
plt.title('Feature importances (y-task, passive funds)', size=20)
plt.xticks(rotation='vertical')
plt.show()

plt.bar(x=importances_y_p.iloc[0:10,0], height=importances_y_p.iloc[0:10,1], color='#087E8B')
plt.title('Feature importances (y-task, passive funds)', size=20)
plt.xticks(rotation='vertical')
plt.show()

# y_a
plt.bar(x=importances_y_a.iloc[0:20,0], height=importances_y_a.iloc[0:20,1], color='#087E8B')
plt.title('Feature importances (y-task, active funds)', size=20)
plt.xticks(rotation='vertical')
plt.show()

plt.bar(x=importances_d_p.iloc[0:20,0], height=importances_d_p.iloc[0:20,1], color='#087E8B')
plt.title('Feature importances (d-task, passive funds)', size=20)
plt.xticks(rotation='vertical')
plt.show()

#d_a
plt.bar(x=importances_d_a.iloc[0:20,0], height=importances_d_a.iloc[0:20,1], color='#087E8B')
plt.title('Feature importances (d-task, active funds)', size=20)
plt.xticks(rotation='vertical')
plt.show()

xgb.plot_importance(xgb_y_p)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# =============================================================================
#%% Checking the variance and sd 
# =============================================================================

df_long_new_p
df_long_new_a

df_long_new_p_var = pd.read_csv(r".../Data/Clean/df_long_p_enc_df.csv", index_col=0)
df_long_new_a_var = pd.read_csv(r".../Data/Clean/df_long_a_enc_df.csv", index_col=0)

df_long_new_p_var = df_long_new_p_var.iloc[:,0:4].drop(['variable'],axis = 1)
df_long_new_a_var = df_long_new_a_var.iloc[:,0:4].drop(['variable'],axis = 1)

df_long_new_p_var_2 = df_long_new_p_var.drop(['year'],axis = 1)
df_long_new_a_var_2 = df_long_new_a_var.drop(['year'],axis = 1)

train_p_2, test_p_2 = train_test_split(df_long_new_p_var_2, 
                                       test_size=0.3,
                                       random_state=1)

train_a_2, test_a_2 = train_test_split(df_long_new_a_var_2,
                                       test_size=0.3, 
                                       random_state=1)

sum_stat_p = df_long_new_p_var.groupby(['year','month']).describe()
sum_stat_p_2 = df_long_new_p_var_2.groupby(['month']).describe()

sum_stat_a = df_long_new_a_var.groupby(['year','month']).describe()
sum_stat_a_2 = df_long_new_a_var_2.groupby(['month']).describe()

sum_stat_p_train = train_p_2.groupby(['month']).describe()
sum_stat_p_test = test_p_2.groupby(['month']).describe()

sum_stat_a_train = train_a_2.groupby(['month']).describe()
sum_stat_a_test = test_a_2.groupby(['month']).describe()

sum_stat_p_train.iloc[:,2].mean()
sum_stat_p_test.iloc[:,2].mean()

sum_stat_a_train.iloc[:,2].mean()
sum_stat_a_test.iloc[:,2].mean()

sum_stat_p_2.iloc[:,2].mean()
sum_stat_a_2.iloc[:,2].mean()