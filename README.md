# DML-in-Finance
Master's degree thesis project using Debiased Machine Learning to estimate treatment effects from economic policy in US funds performance.

How to:

1. We first used data_collection.py to collect all of the data from Yahoo Finance. Data from FRED and other sources was collected manually and is available in the Raw Data folder.
2. FRED data needed some date-time standardization, and interpolation from quarterly to monthly. This is done on the series_disaggregation.py file.
3. Standardized FRED data is joined in the data_join.py file.
4. Fund prices (collected in step 1) are transformed to fund returns (growth rate) using fund_prices_to_returns.py
5. Fund returns and FRED data are combined one fund at a time to run an ANN in the loop_individual_funds.py file. Model logbook.txt keeps a summary of the experiments.
6. VAR.py and VAR_mon.py attempt to build a VAR model.