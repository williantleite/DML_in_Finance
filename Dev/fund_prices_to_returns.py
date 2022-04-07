# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:34:36 2022

@author: gojja och willi
"""
import pandas as pd

active_prices_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\active_prices_df.csv")
passive_prices_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\passive_prices_df.csv")

active_prices_df.iloc[:,3:] = active_prices_df.iloc[:,3:].pct_change(periods=1)
passive_prices_df.iloc[:,3:] = passive_prices_df.iloc[:,3:].pct_change(periods=1)

active_prices_df.to_csv('active_prices_df.csv', index=True)
passive_prices_df.to_csv('passive_prices_df.csv', index=True)
