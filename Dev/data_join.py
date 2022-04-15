# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:12:42 2022

@author: willi och gojja
"""
#Joining all of the data

import pandas as pd
from functools import reduce
from statsmodels.stats.outliers_influence import variance_inflation_factor
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Load data

house_price_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\all_transac_house_price_index_m_df.csv")

anxious_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\anxious_index_m_df.csv")

nrou_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\noncyclical_rate_of_unemployment_m_df.csv")

unrate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\unrate_m_df.csv")

gdp_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\real_gdp_m_df.csv")

consumer_sentiment_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\consumer_sentiment_m_df.csv")

cpi_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\cpi_m_df.csv")

interest_rate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\interest_rate_m_df.csv")

m2_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\m2_m_df.csv")

m3_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\m3_df.csv")

recession_dummy_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\recession_dummy_m_df.csv")

i_rate_growth_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\i_rate_growth_m_df.csv")

#Join the data

variables_list = [anxious_index_df, 
                  consumer_sentiment_df, 
                  cpi_df, 
                  gdp_df, 
                  house_price_index_df, 
                  interest_rate_df, 
                  m2_df, 
                  nrou_df, 
                  recession_dummy_df,
                  unrate_df,
                  i_rate_growth_df]

x_df = reduce(lambda left,right: pd.merge(left, right, on=['year', 'month'], how = "inner"), variables_list)

x_df.columns = ["year", "month", "anxious_index", "consumer_sent", "inflation", "gdp_growth", "hpi", "interest_rate", "m2", "nrou", "recession", "rou", "i_rate_growth"]

x_df.to_csv('x_df.csv', index = False)

#%%
## Correlation and Multicollinearity Analysis

# std_scaler = StandardScaler()
scaler = MinMaxScaler()
# x_df.iloc[:,3:] = std_scaler.fit_transform(x_df.iloc[:,3:].to_numpy())
x_values = x_df.iloc[:, 3:].values
x_values = scaler.fit_transform(x_values)
x_df.iloc[:,3:] = pd.DataFrame(x_values, columns=x_df.iloc[:,3:].columns)

x_corr = x_df.iloc[:, 3:].corr()

from sklearn.decomposition import PCA

pca = PCA(.95)
pca.fit(x_corr)
principalComponents = pca.transform(x_corr)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])
pca.explained_variance_ratio_
pca.n_components_
# def vif(X):

#     # Calculating VIF
#     vif = pd.DataFrame()
#     vif["variables"] = X.columns
#     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#     return(vif)

# vif_fred = vif(x_subset.iloc[:,2:])

# x_subset2 = x_subset.iloc[:,2:].drop(["m2", "inflation"], axis=1)
# vif_fred2 = vif(x_subset2)
# x_subset2_corr = x_subset2.corr()

# x_subset3 = x_subset2.drop("nrou", axis=1)
# vif_fred3 = vif(x_subset3)
# x_subset3_corr = x_subset3.corr()

# x_subset4 = x_subset3.drop("hpi", axis=1)
# vif_fred4 = vif(x_subset4)
# x_subset4_corr = x_subset4.corr()

for i in range(x_df.iloc[2922:,3:].shape[1]):
    plt.plot(x_df.iloc[2922:,(3+i)])
    plt.title(x_df.iloc[2922:,(3+i)].name)
    plt.show()