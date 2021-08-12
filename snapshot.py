from pathlib import Path
import datetime as dt
import glob
import os
import re
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from mylib import market_close_last_next
import requests_cache
# session = requests_cache.CachedSession('yfinance.cache')
# session.headers['User-agent'] = 'my-program/1.0'
# ticker = yf.Ticker('msft aapl goog', session=session)
# # The scraped response will be stored in the cache
# ticker.actions


def sqlite_import(pattern):
    engine = sqlalchemy.create_engine('sqlite:///datasets/pricing.sqlite')
    p = re.compile('datasets/(.*?)\.')
    #for filename in glob.glob('datasets/*.csv.gz'):
    for filename in glob.glob(pattern):
        m = p.match(filename)
        if not m: continue
        symbol = m.group(1)
        pd.read_csv(f'datasets/{symbol}.csv.gz').to_sql(symbol, engine, index=False)
        print(f'INFO: Imported {symbol}')

def junk():
    #with open('sp100.csv') as f:
   #     lines = f.read().splitlines()
    
    Path("datasets").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    symbols = pd.read_csv('sp100.csv', index_col=0)
    for symbol in symbols.Symbol:
        file = "datasets/{}.csv.gz".format(symbol)
        if os.path.isfile(file):
            print(f'INFO: skipping {symbol}')
            continue
        print(f'INFO: downloading {symbol}')
        data = yf.download(symbol, start="2005-01-01", progress=False) #, end="2020-08-22")
        data = yf_df_normalize(data)
        data.to_csv(file)


def yf_df_normalize(df):
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index).normalize()
    if 'adj close' in df: df.drop('adj close',axis=1, inplace=True)
    if 'index' in df: df.drop('index',axis=1, inplace=True)
    return df

def yf_df_update(df):
    in_df = df
    if df is None: return None
    if not df.symbol: return None
    filename = df.filename

    dates = market_close_last_next()
    if not dates[2] > df.iloc[-1].name:
        print(f'filename={df.filename:<25s} symbol={df.symbol:5s} NOOP')
        return None
    
    next_close_ts = df.iloc[-1].name + dt.timedelta(days=1)
    print(f'filename={filename:<25s} symbol={df.symbol:5s} from={next_close_ts}')
    delta_df = yf.download(df.symbol, progress=False, start=next_close_ts.strftime('%Y-%m-%d'))
    delta_df.symbol = df.symbol
    delta_df = yf_df_normalize(delta_df)
    
    df = df.append(delta_df)
    df.filename = in_df.filename
    df.symbol = in_df.symbol
    if not df.empty:
        print(f'filename={df.filename:<25s} symbol={df.symbol:5s} writing')
        df.to_csv(df.filename, index=True)
    else:
        print(f'filename={df.filename:<25s} symbol={df.symbol:5s} OPPS')

    return df

def yf_df_validate(p, filename):
    print(f'filename={filename:<25s} Validating')
    m = p.match(filename)
    if not m: return None
    symbol = m.group(1)
    df = pd.read_csv(filename, parse_dates=True)
    if len(df) == 0:
        print(f'filename={filename:<25s} symbol={symbol:5s} Empty')
        os.remove(filename)
        return None
    if df.shape[1] > 10:
        print(f'filename={filename:<25s} symbol={symbol:5s} Bad shape')
        return None
    df.symbol = symbol
    df.filename = filename
    return df

def update_datasets():
    i = 0
    p = re.compile('datasets/(.*?)\.')
    for filename in glob.glob('datasets/*.csv.gz'):
        i += 1
        df = yf_df_validate(p, filename)
        if df is not None: df = yf_df_normalize(df)
        if df is not None: yf_df_update(df)
    if i > 5:
        return

update_datasets()
#junk()
