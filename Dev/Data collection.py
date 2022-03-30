# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:02:52 2022

@author: willi
"""

### Building Financial dataset from Yahoo Finance

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

passive_list = pd.read_csv(r"C:\Users\willi\Desktop\Passive 1948 csv.csv", 
                           names = ["ticker", "name", "inception_date",
                                    "fund_asset_class_focus", "parent_comp_name",
                                    "fund_industry_focus", "tot_asset_mil"], 
                           header=0)

passive_list["ticker"] = passive_list["ticker"].str.replace(r'\sUS Equity', '')

active_list = pd.read_csv(r"C:\Users\willi\Desktop\Active 4976 csv.csv", 
                          names = ["ticker", "name", "inception_date",
                                   "fund_asset_class_focus", "parent_comp_name",
                                   "fund_industry_focus", "tot_asset_mil"], 
                          header=0)

active_list["ticker"] = active_list["ticker"].str.replace(r'\sUS Equity', '')

passive_tickers = passive_list["ticker"].tolist()
active_tickers = active_list["ticker"].tolist()

passive_df = yf.download(passive_tickers,
                         period = "max",
                         interval = "1mo",
                         progress = True)

passive_df = passive_df["Close"]

active_df = yf.download(active_tickers, 
                        period = "max",
                        interval = "1mo",
                        progress = True)

active_df = active_df["Close"]

passive_df.to_csv('passive_df.csv', index=False)
active_df.to_csv('active_df.csv', index=False)
