### Standardizing some independent variables

"""
Created on Wed Mar 30 16:30:05 2022

@author: gojja och willi

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

def adding_date_variables(df):
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    df["year"], df["month"], = df.index.year, df.index.month
    df.insert(0, "year", df.pop("year"))
    df.insert(1, "month", df.pop("month"))
    return(df)

#Load data

#%% Anxious Index
anxious_index_df = pd.read_excel(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Anxious Index\anxious_index_chart.xlsx")
anxious_index_df = anxious_index_df.iloc[3:,0:3] #Here we exclude the first three lines because they are empty, and also the last column because it is a variable we are not interested in.
anxious_index_df.columns = ["year", "quarter", "anxious_index"]
anxious_index_df = anxious_index_df.astype({"anxious_index": "float64"})
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)
anxious_index_df = anxious_index_df.drop(["year", "quarter", "month"], axis = 1)
anxious_index_df = pad_monthly(anxious_index_df)

anxious_index_df.to_csv('anxious_index_m_df.csv', index=False)  

#%% House Price Index
house_price_index_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\House prices\All-Transactions_House_Price_Index.csv")
house_price_index_df = interpolate_monthly(house_price_index_df)

house_price_index_df.to_csv('all_transac_house_price_index_m_df.csv', index=False)

#%% Natural Rate of Unemployment
nrou_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\Noncyclical_Rate_of_Unemployment.csv")
nrou_df = interpolate_monthly(nrou_df)

nrou_df.to_csv('noncyclical_rate_of_unemployment_m_df.csv', index=False)

#%% Real GDP
gdp_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Real-GDP\Real_GDP.csv")
gdp_df.iloc[:,1] = gdp_df.iloc[:,1].pct_change(periods=1)
gdp_df = interpolate_monthly(gdp_df)

gdp_df.to_csv('real_gdp_m_df.csv', index=False)

#%% Rate of Unemployment
unrate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Unemployment\UNRATE.csv")
unrate_df = adding_date_variables(unrate_df)

unrate_df.to_csv('unrate_m_df.csv', index = False)

#%% Consumer Sentiment
consumer_sentiment_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Consumer Sentiment\Consumer Sentiment.csv")
consumer_sentiment_df = adding_date_variables(consumer_sentiment_df)
consumer_sentiment_df = consumer_sentiment_df.iloc[302:,:].astype(float)

consumer_sentiment_df.to_csv('consumer_sentiment_m_df.csv', index = False)

#%% CPI Inflation
cpi_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\CPI\CPI.csv")
cpi_df = adding_date_variables(cpi_df)

cpi_df.to_csv('cpi_m_df.csv', index = False)

#%% Interest rate
interest_rate_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\FED FUND\FEDFUNDS.csv")
interest_rate_df = adding_date_variables(interest_rate_df)

interest_rate_df.to_csv('interest_rate_m_df.csv', index = False)

#%% M2 Money Supply
m2_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Money supply\M2.csv")
m2_df = adding_date_variables(m2_df)

m2_df.to_csv('m2_m_df.csv', index = False)

#%% Recession indicator
recession_dummy_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\Recession\Recession_Indicators.csv")
recession_dummy_df = adding_date_variables(recession_dummy_df)

recession_dummy_df.to_csv('recession_dummy_m_df.csv', index = False)

#%% Interest rate growth rate
i_rate_growth_df = interest_rate_df.copy()
i_rate_growth_df.iloc[:,2:] = i_rate_growth_df.iloc[:, 2:].pct_change(periods=1)

i_rate_growth_df.to_csv('i_rate_growth_m_df.csv', index = False)

#%% For daily disagregation:
    
# def date_addition(df):
#     df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
#     df = df.set_index("DATE").resample("M").interpolate()
#     df["year"], df["month"] = df.index.year, df.index.month
#     df.insert(0, "year", df.pop("year"))
#     df.insert(1, "month", df.pop("month"))
#     return(df)

# def interpolate_daily(df):
#     df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("D")
#     df = df.set_index("DATE").resample("D").interpolate()
#     df["year"], df["month"], df["day"] = df.index.year, df.index.month, df.index.day
#     df.insert(0, "year", df.pop("year"))
#     df.insert(1, "month", df.pop("month"))
#     df.insert(2, "day", df.pop("day"))
#     return(df)

# def pad_daily(df):
#     df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("D")
#     df = df.set_index("DATE").resample("D").pad()
#     df["year"], df["month"], df["day"] = df.index.year, df.index.month, df.index.day
#     df.insert(0, "year", df.pop("year"))
#     df.insert(1, "month", df.pop("month"))
#     df.insert(2, "day", df.pop("day"))
#     return(df)    