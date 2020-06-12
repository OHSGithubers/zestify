import pandas as pd
from pandas_datareader import data as pdr
from datetime import date
import datetime
import yfinance as yf
yf.pdr_override()
import pandas.testing

# Getting Data
ticker = input('Enter the stock you wish to analyze:')
today = date.today()
today = today + datetime.timedelta(days=1)
start_date="2010-01-01"
data_stock = pdr.get_data_yahoo_actions(ticker, start=start_date, end=today)


# Transforming Data
data_stock = data_stock[::-1].reset_index()

# Save Data
data_stock.to_csv("./stonk_data.csv")
