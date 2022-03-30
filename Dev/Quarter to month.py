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
