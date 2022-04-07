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

# def date_addition(df):
#     df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
#     df = df.set_index("DATE").resample("M").interpolate()
#     df["year"], df["month"] = df.index.year, df.index.month
#     df.insert(0, "year", df.pop("year"))
#     df.insert(1, "month", df.pop("month"))
#     return(df)

def interpolate_daily(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("D")
    df = df.set_index("DATE").resample("D").interpolate()
    df["year"], df["month"], df["day"] = df.index.year, df.index.month, df.index.day
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    df.insert(2, "day", df.pop("day"))
    return(df)

def pad_daily(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("D")
    df = df.set_index("DATE").resample("D").pad()
    df["year"], df["month"], df["day"] = df.index.year, df.index.month, df.index.day
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    df.insert(2, "day", df.pop("day"))
    return(df)

#%% Loading the data and transforming the data:
    
#Anxious index

anxious_index_df = pd.read_excel(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Anxious Index\anxious_index_chart.xlsx", sheet_name = "Data")
anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
anxious_index_df = anxious_index_df.astype({"anxious_index": "float64"})
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = anxious_index_df.drop(["year", "quarter", "month"], axis = 1)
anxious_index_df = interpolate_daily(anxious_index_df)

anxious_index_df.to_csv('anxious_index_df.csv', index=False)

#All-transactions House Price Index

all_transac_house_price_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\House prices\All-Transactions_House_Price_Index.csv")

all_transac_house_price_index_df = interpolate_daily(all_transac_house_price_index_df)

all_transac_house_price_index_df.to_csv('all_transac_house_price_index_df.csv', index=False)

#Noncyclical rate of unemployment

noncyclical_rate_of_unemployment_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\Noncyclical_Rate_of_Unemployment.csv")

noncyclical_rate_of_unemployment_df = interpolate_daily(noncyclical_rate_of_unemployment_df)

noncyclical_rate_of_unemployment_df.to_csv('noncyclical_rate_of_unemployment_df.csv', index=False)

#Real GDP

real_gdp_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Real-GDP\Real_GDP.csv")
real_gdp_df.iloc[:,1] = real_gdp_df.iloc[:,1].pct_change(periods=1)
real_gdp_df = interpolate_daily(real_gdp_df)

real_gdp_df.to_csv('real_gdp_df.csv', index=False)

#Unemployment rate
unrate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\UNRATE.csv")

unrate_df = interpolate_daily(unrate_df)

unrate_df.to_csv('unrate_df.csv', index = False)

#Consumer sentiment

consumer_sentiment_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Consumer Sentiment\Consumer Sentiment.csv")

consumer_sentiment_df = consumer_sentiment_df.iloc[302:,:].astype({"UMCSENT": "float64"})
consumer_sentiment_df["DATE"] = pd.to_datetime(consumer_sentiment_df["DATE"])

consumer_sentiment_df = interpolate_daily(consumer_sentiment_df)

consumer_sentiment_df.to_csv('consumer_sentiment_df.csv', index = False)

#Inflation

cpi_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\CPI\CPI.csv")
cpi_df.iloc[:,1] = cpi_df.iloc[:,1].pct_change(periods=1)
cpi_df = interpolate_daily(cpi_df)

cpi_df.to_csv('cpi_df.csv', index = False)

#Interest rate

interest_rate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\FED FUND\FEDFUNDS.csv")

interest_rate_df = pad_daily(interest_rate_df)

interest_rate_df.to_csv('interest_rate_df.csv', index = False)

#M2 Supply

m2_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Money supply\M2.csv")

m2_df = interpolate_daily(m2_df)

m2_df.to_csv('m2_df.csv', index = False)

#M3 Supply

m3_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Money supply\M3.csv")

m3_df = interpolate_daily(m3_df)

m3_df.to_csv('m3_df.csv', index = False)

# Recession dummy

recession_dummy_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Recession\Recession_Indicators.csv")

recession_dummy_df = pad_daily(recession_dummy_df)

recession_dummy_df.to_csv('recession_dummy_df.csv', index = False)

# Interest rate growth rate

i_rate_growth_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\FED FUND\FEDFUNDS.csv")
i_rate_growth_df.iloc[:,1:] = i_rate_growth_df.iloc[:, 1:].pct_change(periods=1)
i_rate_growth_df = pad_daily(i_rate_growth_df)

i_rate_growth_df.to_csv('i_rate_growth_df.csv', index = False)
