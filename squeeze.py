import os
import os.path
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime 
from pathlib import Path

dataframes = {}

def load_symbol(symbol):
    file = f"datasets/{symbol}.csv.gz"
    if os.path.isfile(file):
        df = pd.read_csv(file)
    else:
        start_dt = datetime.datetime.now() - datetime.timedelta(days = 50)
        start_str = start_dt.strftime("%Y-%m-%d")
        df = yf.download(symbol, start=start_str)
        df.to_csv(file,index=True)
    return df

for filename in os.listdir('datasets'):
    symbol = filename.split(".")[0]
    df = load_symbol(symbol)
    if df.empty:
        continue

    df['20sma'] = df['Close'].rolling(window=20).mean()
    df['stddev'] = df['Close'].rolling(window=20).std()
    df['lower_band'] = df['20sma'] - (2 * df['stddev'])
    df['upper_band'] = df['20sma'] + (2 * df['stddev'])

    df['TR'] = abs(df['High'] - df['Low'])
    df['ATR'] = df['TR'].rolling(window=20).mean()

    df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
    df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

    def in_squeeze(df):
        return df['lower_band'] > df['lower_keltner'] and df['upper_band'] < df['upper_keltner']

    df['squeeze_on'] = df.apply(in_squeeze, axis=1)

    if df.iloc[-3]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
        print(f"{symbol} ON")
    elif df.iloc[-3]['squeeze_on'] and df.iloc[-2]['squeeze_on'] and df.iloc[-1]['squeeze_on']:
        print(f"{symbol} REST")

    # save all dataframes to a dictionary
    # we can chart individual names below by calling the chart() function
    dataframes[symbol] = df


def chart(df):
    candlestick = go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])
    upper_band = go.Scatter(x=df['Date'], y=df['upper_band'], name='Upper Bollinger Band', line={'color': 'red'})
    lower_band = go.Scatter(x=df['Date'], y=df['lower_band'], name='Lower Bollinger Band', line={'color': 'red'})

    upper_keltner = go.Scatter(x=df['Date'], y=df['upper_keltner'], name='Upper Keltner Channel', line={'color': 'blue'})
    lower_keltner = go.Scatter(x=df['Date'], y=df['lower_keltner'], name='Lower Keltner Channel', line={'color': 'blue'})

    fig = go.Figure(data=[candlestick, upper_band, lower_band, upper_keltner, lower_keltner])
    fig.layout.xaxis.type = 'category'
    fig.layout.xaxis.rangeslider.visible = False
    fig.show()

df = dataframes['GOOGL']
chart(df)