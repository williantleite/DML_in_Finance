# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:12:28 2022

@author: gojja och willi
"""

import pandas as pd

FEDFUNDS_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Raw Data\Other Variables\FED FUND\FEDFUNDS.csv")

FEDFUNDS_df.iloc[:,1:] = FEDFUNDS_df.iloc[:,1:].pct_change(periods=1)

FEDFUNDS_df.to_csv('FEDFUNDS_growth_rate.csv', index=False)


