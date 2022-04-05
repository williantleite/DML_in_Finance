# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:30:05 2022

@author: willi och gojja
"""

### Standardizing some independent variables

"""
Some of the variables selected were only available with quarterly information.
Here we standardize them to monthly by repeating the data for the quarter along
the months within that quarter.
""" 

import pandas as pd

#Quarterly data to monthly data by repeating the quarterly values over the months

def date_addition(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df = df.set_index("DATE").resample("M").interpolate()
    df["year"], df["month"] = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

#%% Loading the data and transforming the data:
    
#Anxious index

anxious_index_df = pd.read_excel(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Anxious Index\anxious_index_chart.xlsx", sheet_name = "Data")
anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = date_addition(anxious_index_df)
anxious_index_df.pop("quarter")

anxious_index_df.to_csv('anxious_index_df.csv', index=False)

#All-transactions House Price Index

all_transac_house_price_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\House prices\All-Transactions_House_Price_Index.csv")

all_transac_house_price_index_df = date_addition(all_transac_house_price_index_df)

all_transac_house_price_index_df.to_csv('all_transac_house_price_index_df.csv', index=False)

#Noncyclical rate of unemployment

noncyclical_rate_of_unemployment_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\Noncyclical_Rate_of_Unemployment.csv")

noncyclical_rate_of_unemployment_df = date_addition(noncyclical_rate_of_unemployment_df)

noncyclical_rate_of_unemployment_df.to_csv('noncyclical_rate_of_unemployment_df.csv', index=False)

# Real GDP

real_gdp_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Real-GDP\Real_GDP.csv")
real_gdp_df.iloc[:,1] = real_gdp_df.iloc[:,1].pct_change(periods=1)
real_gdp_df = date_addition(real_gdp_df)

real_gdp_df.to_csv('real_gdp_df.csv', index=False)
