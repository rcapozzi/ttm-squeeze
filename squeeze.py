#!/usr/bin/env python

import os
import os.path
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objects as go
import datetime 
from pathlib import Path

dataframes = {}

def symbolsfor(str):
    if str == 'SPX':
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers =  [i.replace('.','-') for i in df.Symbol.to_list()]
    elif str == 'NDX':
        df = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100', attrs = {'id': 'constituents'})[0]
        tickers = df.Ticker.values
    return tickers

def load_symbol(symbol):
    file = f"datasets/{symbol}.csv.gz"
    if os.path.isfile(file):
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
    else:
        start_dt = datetime.datetime.now() - datetime.timedelta(days = 50)
        start_str = start_dt.strftime("%Y-%m-%d")
        df = yf.download(symbol, progress=False, period='3mo') #start=start_str)
        df.columns = df.columns.str.lower()
        df.to_csv(file,index=True)
    df.name = symbol
    return df

def is_ema_stacked(df):
    avg = [
        df.ta.ema(8).iloc[-1],
        df.ta.ema(21).iloc[-1],
        df.ta.ema(34).iloc[-1]
    ]
    ema = pd.concat({
        'ema8': df.ta.ema(8),
        'ema21': df.ta.ema(21),
        'ema34': df.ta.ema(34)
        }, axis=1)
    ema['stack_bull'] = (ema.ema8 > ema.ema21) & (ema.ema21 > ema.ema34)
    ema['stack_bear'] = (ema.ema8 < ema.ema21) & (ema.ema21 < ema.ema34)
    return avg[0] > avg[1] and avg[1] > avg[2]

def inspect_squeeze(symbol, df=None):
    if not df: 
        df = load_symbol(symbol)
    if df.empty:
        return None

    status_spro = 0
    status_ema = 0
    status_adx = 0
    status_buy_zone = 0

    spro = df.ta.squeeze_pro()
    if spro.iloc[-4:].SQZPRO_ON_NORMAL.sum() == 4:
        status_spro = 1

    if is_ema_stacked(df):
        status_ema = 1

    adx = df.ta.adx().ADX_14.iloc[-1]
    if adx < 20.0:
        status_adx = 1

    if df.ta.ema(21).iloc[-1] * 1.05 >  df.iloc[-1].close:
        status_buy_zone = 1
    bar = df.iloc[-1]

    rsi = df.ta.rsi().iloc[-1]
    atr = df.ta.atr().iloc[-1]
    sell1= bar.close + (atr*2)
    est_ret = (atr*2)/bar.close*100
    str = f'atr={atr:05.2f} rsi={rsi:5.2f} close={bar.close:6.2f} sell1={sell1:6.2f} est_ret={est_ret:05.2f}'

    status_verdict = status_spro + status_ema + status_adx + status_buy_zone
    print(f"{symbol:4s} spro={status_spro} ema={status_ema} adx={adx:2.0f} buy_zone={status_buy_zone} verdict={status_verdict} {str}")
    return df


for filename in os.listdir('datasets'):
    symbol = filename.split(".")[0]
    inspect_squeeze(symbol)
    continue

    df = load_symbol(symbol)
    if df.empty:
        continue
    inspect_squeeze(symbol)

    # df['20sma'] = df['close'].rolling(window=20).mean()
    # df['stddev'] = df['close'].rolling(window=20).std()
    # df['lower_band'] = df['20sma'] - (2 * df['stddev'])

    # def in_squeeze(df):
    #     return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

    # df['squeeze_on'] = df.apply(in_squeeze, axis=1)

    # save all dataframes to a dictionary
    # we can chart individual names below by calling the chart() function
    dataframes[symbol] = df
    status_spro = 0
    status_ema = 0
    status_adx = 0
    status_buy_zone = 0

    spro = df.ta.squeeze_pro()
    if spro.iloc[-4:].SQZPRO_ON_NORMAL.sum() == 4:
        status_spro = 1

    if is_ema_stacked(df):
        status_ema = 1

    adx = df.ta.adx().ADX_14.iloc[-1]
    if adx < 20.0:
        status_adx = 1

    if df.ta.ema(21).iloc[-1] * 1.05 >  df.iloc[-1].close:
        status_buy_zone = 1
    bar = df.iloc[-1]

    rsi = df.ta.rsi().iloc[-1]
    atr = df.ta.atr().iloc[-1]
    sell1= bar.close + (atr*2)
    est_ret = (atr*2)/bar.close*100
    str = f'atr={atr:05.2f} rsi={rsi:5.2f} close={bar.close:6.2f} sell1={sell1:6.2f} est_ret={est_ret:05.2f}'

    status_verdict = status_spro + status_ema + status_adx + status_buy_zone
    print(f"{symbol:4s} spro={status_spro} ema={status_ema} adx={adx:2.0f} buy_zone={status_buy_zone} verdict={status_verdict} {str}")

    # if df.iloc[-3]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
    #     print(f"{symbol} ON")
    # elif df.iloc[-3]['squeeze_on'] and df.iloc[-2]['squeeze_on'] and df.iloc[-1]['squeeze_on']:
    #     print(f"{symbol} REST")

def getSignals_v(df):
    df['rsi'] = df.ta.rsi()
    df['sma'] = df.ta.sma(200)
    df.dropna(inplace=True)
    
    stop_loss_pct = 4.0
    max_lookahead = 11
    cutoff = df.tail(max_lookahead).index.min()
    signals = df.loc[(df.close > df.sma) & (df.rsi < 30) & (df.index < cutoff)]
    signals = signals[signals.index < df.index.max() - dt.timedelta(days=15)]
    bdates, sdates = [], []
    trades = []
    for idx, row in signals.iterrows():
        iloc = df.index.get_loc(idx)
        buy_day = df.iloc[iloc+1]
        sell_stop = buy_day.open * (1.0 - (stop_loss_pct/100)
        bdates.append(buy_day.name)
        sell_day = None
        for j in range(1,max_lookahead):
            this_day = df.iloc[iloc+j]
            sell_day = df.iloc[iloc+1+j+1]
            if this_day.rsi > 40: break
        sdates.append(sell_day.name)
        trades.append(buy_day.name, buy_day.open, )
        #print(f'i:{i:03d} iloc:{iloc:05d} buy_on:{buy_day.name} sell_on:{sell_day.name}')
    df = pd.DataFrame(trades, columns=['A', 'B', 'C'])
    return bdates, sdates


def chart(df):
    candlestick = go.Candlestick(x=df['Date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'])
    upper_band = go.Scatter(x=df['Date'], y=df['upper_band'], name='Upper Bollinger Band', line={'color': 'red'})
    lower_band = go.Scatter(x=df['Date'], y=df['lower_band'], name='Lower Bollinger Band', line={'color': 'red'})

    upper_keltner = go.Scatter(x=df['Date'], y=df['upper_keltner'], name='Upper Keltner Channel', line={'color': 'blue'})
    lower_keltner = go.Scatter(x=df['Date'], y=df['lower_keltner'], name='Lower Keltner Channel', line={'color': 'blue'})

    fig = go.Figure(data=[candlestick, upper_band, lower_band, upper_keltner, lower_keltner])
    fig.layout.xaxis.type = 'category'
    fig.layout.xaxis.rangeslider.visible = False
    fig.show()

# df = dataframes['YUM']
# chart(df)