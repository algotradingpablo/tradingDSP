import pandas as pd
import mplfinance as mpf

daily = pd.read_csv("five_years_daily_eurusd.csv", index_col=0) 
weekly = pd.read_csv("five_years_weekly_eurusd.csv", index_col=0)

daily.index = pd.to_datetime(daily.index)
weekly.index = pd.to_datetime(weekly.index)

mpf.plot(daily, type='candle', style='charles', title='USD/EUR daily', ylabel='Price ($)')
mpf.plot(weekly, type='candle', style='charles', title='USD/EUR weekly', ylabel='Price ($)')