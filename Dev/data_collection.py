# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:02:52 2022

@author: willi och gojja
"""

### Building Financial dataset from Yahoo Finance

import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
import numpy as np

passive_list = pd.read_csv(r"...\Data\Raw Data\Funds\List 1948 Passive Funds.csv", 
                           names = ["ticker", "name", "inception_date",
                                    "fund_asset_class_focus", "parent_comp_name",
                                    "fund_industry_focus", "tot_asset_mil"], 
                           header = 0)

passive_list["ticker"] = passive_list["ticker"].str.replace(r'\sUS Equity', '')

active_list = pd.read_csv(r"...\Data\Raw Data\Funds\List 4976 Active Funds.csv", 
                          names = ["ticker", "name", "inception_date",
                                   "fund_asset_class_focus", "parent_comp_name",
                                   "fund_industry_focus", "tot_asset_mil"], 
                          header = 0)

active_list["ticker"] = active_list["ticker"].str.replace(r'\sUS Equity', '')

passive_tickers = passive_list["ticker"].tolist()
active_tickers = active_list["ticker"].tolist()

passive_df = yf.download(passive_tickers,
                         start = "1986-01-01",
                         end = "2022-01-01",
                         interval = "1mo",
                         progress = True)

passive_df = passive_df["Adj Close"]

passive_df = passive_df.dropna(how = "all", axis = 1)
passive_df = passive_df.dropna(how = "all", axis = 0)
passive_df["year"], passive_df["month"] = passive_df.index.year, passive_df.index.month
passive_df.insert(0, "year", passive_df.pop("year"))
passive_df.insert(1, "month", passive_df.pop("month"))

passive_df.to_csv('passive_prices_m_df.csv', index=False)

active_df = yf.download(active_tickers, 
                        start = "1986-01-01",
                        end = "2022-01-01",
                        interval = "1mo",
                        progress = True)

active_df = active_df["Adj Close"]

active_df = active_df.dropna(how="all", axis=1)
active_df = active_df.dropna(how="all", axis=0)
active_df["year"], active_df["month"] = active_df.index.year, active_df.index.month
active_df.insert(0, "year", active_df.pop("year"))
active_df.insert(1, "month", active_df.pop("month"))

active_df.to_csv('active_prices_m_df.csv', index=False)
