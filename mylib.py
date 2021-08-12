# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 05:10:39 2021

@author: R_Capozzi
"""

import datetime as dt
import pandas as pd
import pandas_market_calendars as mcal
import functools

# Assume the date on a dataframe is the date of the market close.
# market close is typically 4pm US/Eastern
# ts.round(freq='H')
@functools.lru_cache(maxsize=5)
def market_close_last_next(when_ts=None):
    if when_ts is None:
        when_ts = pd.Timestamp.utcnow()
    if when_ts.tzname() == None:
        when_ts = when_ts.tz_localize(tz='US/Eastern')
    
    nyse = mcal.get_calendar('NYSE')
    dates = nyse.schedule(start_date=when_ts - dt.timedelta(days=5), end_date=when_ts).market_close
    last_close_ts = dates[dates < when_ts].max()
    next_close_ts = dates[dates > when_ts].min()
    last_close_dt = last_close_ts.normalize().tz_convert(None) # Midnight w/o 
    return (last_close_ts, next_close_ts, last_close_dt)
    

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        #Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]

def df_need_update(df):
    return market_close_last_next()[2] > df.iloc[-1].name
    
# dates = market_close_last_next()
# print(dates)

# df = pd.read_csv('datasets/AAPL.csv.gz', index_col=0, parse_dates=True)
# need_update = dates[2] > df.iloc[-1].name
# print(f'need_update: {need_update}')
