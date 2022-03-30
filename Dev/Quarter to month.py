# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:30:05 2022

@author: willi
"""

### Standardizing some independent variables

"""
Some of the variables selected were only available with quarterly information.
Here we standardize them to monthly by repeating the data for the quarter along
the months within that quarter.
""" 

import pandas as pd

anxious_index_df = pd.read_excel(r"C:\Users\willi\Documents\Python\Thesis\Data\Other Variables\Anxious Index\anxious_index_chart.xlsx", sheet_name = "Data")

anxious_index_df.columns = ["year", "quarter", "anxious_index", "recess"]

anxious_index_df = anxious_index_df.iloc[3:,0:3]
month_dict = {1:"01", 2:"04", 3:"07", 4:"10"}
anxious_index_df["month"] = anxious_index_df["quarter"].apply(lambda x:month_dict[x])
anxious_index_df["year"] = anxious_index_df["year"].astype(str)
anxious_index_df["DATE"] = anxious_index_df[["year", "month"]].agg("-".join, axis=1)

def date_addition(df):
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M")
    df = df.set_index("DATE").resample("M").pad(limit=3)
    df["month"] = df.index.month
    df.insert(1, "month", df.pop("month"))
    return(df)

anxious_index_df = date_addition(anxious_index_df)
anxious_index_df.pop("quarter")
