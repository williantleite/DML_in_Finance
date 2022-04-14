# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:58:47 2022

@author: gojja
"""

import pandas as pd

#Load data

house_price_index_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/All-Transactions_House_Price_Index_montly.csv")

anxious_index_df = pd.read_excel(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/Survey of Professional Forecasters/anxious_index_chart.xlsx")

nrou_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/Noncyclical_Rate_of_Unemployment_montly.csv")

unrate_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/UNRATE.csv")

gdp_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/Real_GDP/Real_GDP_montly.csv")

consumer_sentiment_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/Consumer Sentiment.csv")

cpi_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/CPI.csv")

interest_rate_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/FEDFUNDS.csv")

m2_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/M2.csv")

recession_dummy_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/Recession_Indicators.csv")

i_rate_growth_df = pd.read_csv(r"C:/Users/gojja/OneDrive/Skrivbord/Skolarbete/Master/Thesis/Data/Macro economic data/FRED/FEDFUNDS_growth_rate.csv")




### Standardizing some independent variables

"""
Some of the variables selected were only available with quarterly information.
Here we standardize them to monthly by repeating the data for the quarter along
the months within that quarter.
""" 


#Quarterly data to monthly data by repeating the quarterly values over the months

# def date_addition(df):
#     df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
#     df = df.set_index("DATE").resample("M").interpolate()
#     df["year"], df["month"] = df.index.year, df.index.month
#     df.insert(0, "year", df.pop("year"))
#     df.insert(1, "month", df.pop("month"))
#     return(df)

def interpolate_monthly(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df = df.set_index("DATE").resample("M").interpolate()
    df["year"], df["month"] = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

def pad_monthly(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df = df.set_index("DATE").resample("M").pad()
    df["year"], df["month"] = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

#%% Loading the data and transforming the data:
    
#Anxious index

anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
anxious_index_df = anxious_index_df.astype({"anxious_index": "float64"})
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = anxious_index_df.drop(["year", "quarter", "month"], axis = 1)
anxious_index_df = interpolate_monthly(anxious_index_df)





def adding_date_variables(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    df["year"], df["month"], df["day"] = df.index.year, df.index.month, df.index.day
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    df.insert(2, "day", df.pop("day"))
    return(df)

consumer_sentiment_df = adding_date_variables(consumer_sentiment_df)
cpi_df = adding_date_variables(cpi_df)
interest_rate_df = adding_date_variables(interest_rate_df)
i_rate_growth_df = adding_date_variables(i_rate_growth_df)
m2_df = adding_date_variables(m2_df)
recession_dummy_df = adding_date_variables(recession_dummy_df)
unrate_df = adding_date_variables(unrate_df)


consumer_sentiment_df
# Joining the data together:
x_mon = consumer_sentiment_df.merge(house_price_index_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(cpi_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(interest_rate_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(i_rate_growth_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(m2_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(nrou_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(recession_dummy_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(unrate_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(gdp_df, left_on=('year', 'month'), right_on = ('year', 'month'))
x_mon = x_mon.merge(anxious_index_df, left_on=('year', 'month'), right_on = ('year', 'month'))

x_mon.rename(columns = {'USSTHPI':'hpi', 
                            'UMCSENT':'c_sent',
                            'CPIAUCSL':'cpi',
                            'FEDFUNDS_x':'fedfund',
                            'FEDFUNDS_y':'fedfund_gr',
                            'M2SL':'m2',
                            'MABMM301USM189S':'m3',
                            'NROU':'nrou',
                            'USREC':'rec',
                            'UNRATE':'unrate',
                            'GDPC1':'real_gdp',
                            'anxious_index':'anxious'},
                 inplace = True)

x_mon = x_mon.iloc[131:,:]
x_mon["c_sent"] = x_mon.c_sent.astype(float)

x_mon.to_csv('x_mon_df.csv', index=False)


