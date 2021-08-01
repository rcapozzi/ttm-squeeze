from pathlib import Path
import datetime as dt
import glob
import os
import re
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

import requests_cache
# session = requests_cache.CachedSession('yfinance.cache')
# session.headers['User-agent'] = 'my-program/1.0'
# ticker = yf.Ticker('msft aapl goog', session=session)
# # The scraped response will be stored in the cache
# ticker.actions


def junk():
    with open('symbols.csv') as f:
        lines = f.read().splitlines()
    Path("datasets").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    for symbol in lines:
        file = "datasets/{}.csv.gz".format(symbol)
        if os.path.isfile(file):
            print(f'INFO: skipping {symbol}')
            continue
        print(f'INFO: downloading {symbol}')
        data = yf.download(symbol, start="2005-01-01", progress=False) #, end="2020-08-22")
        data.to_csv(file)


def yf_df_normalize(symbol, df):
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index).normalize()
    if 'adj close' in df: df.drop('adj close',axis=1, inplace=True)
    if 'index' in df: df.drop('index',axis=1, inplace=True)
    df.symbol = symbol
    return df

def yf_df_update(symbol, filename):
    print(f'symbol={symbol:5s} reading...')
    df = pd.read_csv(filename, parse_dates=True)
    if len(df) == 0:
        print(f'symbol={symbol:5s} removing empty file')
        os.remove(filename)
        return None
    if df.shape[1] > 10:
        print(f'symbol={symbol:5s} Wrong shape')
        return None
    df = yf_df_normalize(symbol, df)
    
    symbol = df.symbol
    last_ts = df.iloc[-1].name
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=last_ts, end_date=last_ts)
    last_ts = schedule.loc[last_ts].market_close
    next_close_ts = last_ts + dt.timedelta(days=1)

    # On weekends, the schedule is an empty DF, So get the last several days and take the last
    now_ts = pd.Timestamp.utcnow()
    schedule = nyse.schedule(start_date=now_ts - dt.timedelta(days=5), end_date=now_ts)[-1:]
    last_market_close_ts = nyse.schedule(start_date=now_ts - dt.timedelta(days=5), end_date=now_ts).market_close.iloc[-1]

    if next_close_ts >= last_market_close_ts:
        return False
    # TODO: How does this handle weekends and mid day
    # end is not included
    print(f'symbol={symbol:5s} from={next_close_ts}')
    delta_df = yf.download(symbol, progress=False, start=next_close_ts.strftime('%Y-%m-%d'))
    delta_df = yf_df_normalize(symbol, delta_df)
    
    df = df.append(delta_df)
    df.symbol = symbol
    df.to_csv(filename, index=True)
    return df

def update_datasets():
    p = re.compile('datasets/(.*?)\.')
    for filename in glob.glob('datasets/*.csv.gz'):
        m = p.match(filename)
        if not m: continue
        symbol = m.group(1)
        df = yf_df_update(symbol, filename)

update_datasets()

