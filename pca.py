# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler



prices = pd.read_csv('datasets/AAPL.csv.gz', index_col=0)
# prices['log_ret'] = np.log(prices.close / prices.close.shift(1))

CustomStrategy = ta.Strategy(
    name="Momo and Volatility",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        {"kind": "sma", "length": 8},
        {"kind": "sma", "length": 16},        
        {"kind": "sma", "length": 32},
        {"kind": "sma", "length": 64},
        {"kind": "sma", "length": 128},
        # For MOBO bands, std =  0.8, bars = 10
#        {"kind": "bbands", "length": 20},
#        {"kind": "bbands", "length": 10},
        {"kind": "rsi"},
#        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "log_return"},
#        {"kind": "percent_return"},
    ]
)

def normalize_features(df :pd.DataFrame, features):
    for e in features:
        s = df[e] / df.close - 1
        del df[e]
        df[e] = s
    return df

to_norm = ['SMA_8', 'SMA_16', 'SMA_32', 'SMA_64', 'SMA_128' ]
features = ['RSI', 'SMA_8', 'SMA_16', 'SMA_32', 'SMA_64', 'SMA_128' ]

prices = pd.read_csv('datasets/AAPL.csv.gz', index_col=0)
prices.ta.strategy(CustomStrategy)
prices.ta.strategy(CustomStrategy)

df = normalize_features(prices, to_norm)
df = label_data(prices)

x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)

y = df.loc[:,['target']].values

