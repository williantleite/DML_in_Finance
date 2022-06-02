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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

#Load data

house_price_index_df = pd.read_csv(r"F:\Thesis\Data\Clean\all_transac_house_price_index_m_df.csv")

def pad_monthly(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df = df.set_index("DATE").resample("M").ffil()
    df["year"], df["month"] = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

anxious_index_df = pd.read_csv(r"F:\Thesis\Data\Clean\anxious_index_m_df.csv")

anxious_index_df = pad_monthly(anxious_index_df)

nrou_df = pd.read_csv(r"F:\Thesis\Data\Clean\noncyclical_rate_of_unemployment_m_df.csv")

unrate_df = pd.read_csv(r"F:\Thesis\Data\Clean\unrate_m_df.csv")

gdp_df = pd.read_csv(r"F:\Thesis\Data\Clean\real_gdp_m_df.csv")

consumer_sentiment_df = pd.read_csv(r"F:\Thesis\Data\Clean\consumer_sentiment_m_df.csv")

cpi_df = pd.read_csv(r"F:\Thesis\Data\Clean\cpi_m_df.csv")

interest_rate_df = pd.read_csv(r"F:\Thesis\Data\Clean\interest_rate_m_df.csv")

m2_df = pd.read_csv(r"F:\Thesis\Data\Clean\m2_m_df.csv")

m3_df = pd.read_csv(r"F:\Thesis\Data\Clean\m3_df.csv")

recession_dummy_df = pd.read_csv(r"F:\Thesis\Data\Clean\recession_dummy_m_df.csv")

i_rate_growth_df = pd.read_csv(r"F:\Thesis\Data\Clean\i_rate_growth_m_df.csv")

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

y_p = pd.read_csv(r"F:\Thesis\Data\Clean\passive_prices_m_df.csv", index_col=0)
y_a = pd.read_csv(r"F:\Thesis\Data\Clean\active_prices_m_df.csv", index_col=0)

agg_y = y_p.iloc[:,:2]
agg_y["mean"] = y_p.iloc[:,2:].mean(axis=1)
agg_y = agg_y[np.isfinite(agg_y).all(axis = 1)]

agg_y_a = y_a.iloc[:,:2]
agg_y_a['mean'] = y_a.iloc[:,2:].mean(axis=1)
agg_y_a = agg_y_a[np.isfinite(agg_y_a).all(axis=1)]

x_p_c = pd.merge(agg_y, x_df, on=['year', 'month'], how = 'inner')
x_a_c = pd.merge(agg_y_a, x_df, on = ['year', 'month'], how = 'inner')

#%%
## Correlation and Multicollinearity Analysis

cmap = sn.diverging_palette(250, 15, s=75, l=40, n=9, center='light', as_cmap=True)

scaler = StandardScaler()
# x_df.iloc[:,3:] = std_scaler.fit_transform(x_df.iloc[:,3:].to_numpy())
x_values = x_df.iloc[:, 3:].values
x_values = scaler.fit_transform(x_values)
x_df.iloc[:,3:] = pd.DataFrame(x_values, columns=x_df.iloc[:,3:].columns)

x_corr = x_df.iloc[:, 3:].corr()

x_p_c_values = x_p_c.iloc[:,2:].values
x_p_c_values = scaler.fit_transform(x_p_c_values)
x_p_c.iloc[:,2:] = pd.DataFrame(x_p_c_values, columns=x_p_c.iloc[:,2:].columns)

x_p_c_corr = x_p_c.iloc[:,2:].corr()
mask_p_c = np.triu(np.ones_like(x_p_c_corr, dtype=bool))
plt.title("Passive Set Correlation Matrix")
sn.heatmap(x_p_c_corr, mask=mask_p_c, center = 0, annot=True, fmt='.1f', square = True, cmap = cmap)

x_a_c_values = x_a_c.iloc[:,2:].values
x_a_c_values = scaler.fit_transform(x_a_c_values)
x_a_c.iloc[:,2:] = pd.DataFrame(x_a_c_values, columns=x_a_c.iloc[:,2:].columns)

x_a_c_corr = x_a_c.iloc[:,2:].corr()
mask_a_c = np.triu(np.ones_like(x_a_c_corr, dtype=bool))
plt.title("Active Set Correlation Matrix")
sn.heatmap(x_a_c_corr, mask=mask_a_c, center = 0, annot=True, fmt='.1f', square = True, cmap = cmap)

from sklearn.decomposition import PCA

pca = PCA(.95)
pca.fit(x_p_c_corr)
principalComponents = pca.transform(x_p_c_corr)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
principalDf.index = x_p_c_corr.columns
pca.explained_variance_ratio_
pca.n_components_
plot = sn.heatmap(principalDf, annot = True, fmt = '.1f', cmap=cmap)
plot.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plot.set_title('PCA Passive Set')

pca = PCA(.95)
pca.fit(x_a_c_corr)
principalComponents = pca.transform(x_a_c_corr)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3', 'PC4'])
principalDf.index = x_a_c_corr.columns
pca.explained_variance_ratio_
pca.n_components_
plot = sn.heatmap(principalDf, annot = True, fmt = '.1f', cmap=cmap)
plot.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plot.set_title('PCA Active Set')

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

fig, axs = plt.subplots(3, 4, figsize=(20, 10))
axe = axs.ravel()
for i in range(x_a_c.iloc[:,2:].shape[1]):
    x_a_c.iloc[:,(2+i)].plot(ax=axe[i], title=x_a_c.iloc[:,(2+i)].name)
fig.suptitle('Active set data', fontsize=16, y=.93)
plt.show()

#Average Y over all Y distributions
y_a.iloc[:,2:102].plot(color='lightgray', legend = None)
x_a_c.iloc[:,2].plot(label = "Average Y", legend = True, title="Average Y superimposed over first 100 funds")
