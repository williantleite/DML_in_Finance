# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:34:36 2022

@author: gojja och willi
"""
import pandas as pd	

active_prices_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\active_prices_m_df.csv")
passive_prices_df = pd.read_csv(r"C:\Users\willi\Documents\Python\Thesis\Data\Clean\passive_prices_m_df.csv")

active_prices_df.iloc[:,2:] = active_prices_df.iloc[:,2:].pct_change(periods=1)
passive_prices_df.iloc[:,2:] = passive_prices_df.iloc[:,2:].pct_change(periods=1)

active_prices_df.to_csv('active_prices_m_df.csv', index=True)
passive_prices_df.to_csv('passive_prices_m_df.csv', index=True)

p1 =active_prices_df.iloc[:,6].plot(title=active_prices_df.iloc[:,6].name)

p2 = active_prices_df.loc[:,'AMAX'].plot(title=active_prices_df.loc[:,'AMAX'].name, xlim=[0,9076])