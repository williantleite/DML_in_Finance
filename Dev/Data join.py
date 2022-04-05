# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:12:42 2022

@author: willi och gojja
"""
#Joining all of the data

import pandas as pd
from functools import reduce
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

#Load data

active_prices_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\active_prices_df.csv")

l_active_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\log_active_df.csv")

passive_prices_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\passive_prices_df.csv")

l_passive_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\log_passive_df.csv")

house_price_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\all_transac_house_price_index_df.csv")

anxious_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\anxious_index_df.csv")

nrou_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\noncyclical_rate_of_unemployment_df.csv")

unrate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\UNRATE.csv")

gdp_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\real_gdp_df.csv")

consumer_sentiment_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Consumer Sentiment\Consumer Sentiment.csv")

cpi_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\CPI\CPI.csv")

interest_rate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\FED FUND\FEDFUNDS.csv")

m2_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Money supply\M2.csv")

m3_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Money supply\M3.csv")

recession_dummy_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Recession\Recession_Indicators.csv")

#Standardize date format for the variables that aren't standardized

def date_reformat(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    df["year"], df["month"] = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

consumer_sentiment_df = date_reformat(consumer_sentiment_df)
cpi_df = date_reformat(cpi_df)
interest_rate_df = date_reformat(interest_rate_df)
m2_df = date_reformat(m2_df)
m3_df = date_reformat(m3_df)
recession_dummy_df = date_reformat(recession_dummy_df)
unrate_df = date_reformat(unrate_df)

#Join the data

variables_list = [anxious_index_df, 
                  consumer_sentiment_df, 
                  cpi_df, 
                  gdp_df, 
                  house_price_index_df, 
                  interest_rate_df, 
                  m2_df, 
                  m3_df, 
                  nrou_df, 
                  recession_dummy_df,
                  unrate_df]

x_df = reduce(lambda left,right: pd.merge(left, right, on=['year', 'month'], how = "inner"), variables_list)

x_df.columns = ["year", "month", "anxious_index", "consumer_sent", "inflation", "gdp", "hpi", "interest_rate", "m2", "m3", "nrou", "recession", "rou"]

#Standardized before joining
x_subset = x_df.iloc[132:,:].astype(float)
std_scaler = StandardScaler()
x_subset.iloc[:,2:] = std_scaler.fit_transform(x_subset.iloc[:,2:].to_numpy())
x_subset = pd.DataFrame(x_subset, columns=x_subset.columns)

#Finally, join the data

fl_active_df = pd.merge(l_active_df, x_subset, on = ['year', 'month'], how = "inner")

fl_active_df.to_csv("fl_active_df.csv", index=False)

fl_passive_df= pd.merge(l_passive_df, x_subset, on = ['year', 'month'], how = "inner")

fl_passive_df.to_csv("fl_passive_df.csv", index=False)

#%%
## Correlation and Multicollinearity Analysis

x_corr = x_subset.iloc[:, 2:].corr()

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
