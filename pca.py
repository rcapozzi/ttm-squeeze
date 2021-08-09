#!/usr/bin/python3
# # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler


def add_features(df :pd.DataFrame):
    df = df.copy()
    features = []
    df.ta.percent_return(1,append=True)
    df.ta.percent_return(5,append=True)
    df.ta.percent_return(10,append=True)

    smas = [9, 21, 34, 50, 100, 150]
    for i in smas:
        key = 'SMA_' + str(i)
        df[key] = df.ta.sma(i) / df.close - 1
        features.append(key)

    df['SMA_STACKED'] = 0.5
    df.loc[(df.SMA_STACKED == 0.5) & (df.SMA_9 > df.SMA_21) & (df.SMA_21 > df.SMA_50) & (df.SMA_50 > df.SMA_100) , 'SMA_STACKED']  = 1
    df.loc[(df.SMA_STACKED == 0.5) & (df.SMA_9 < df.SMA_21) & (df.SMA_21 < df.SMA_50) & (df.SMA_50 < df.SMA_100) , 'SMA_STACKED']  = 0

    for i in [9, 14]:
        key = 'RSI_' + str(i)
        df[key] = df.ta.rsi(i) / 100

    df['ADX'] = df.ta.adx().ADX_14/100
    df['ADX_TREND'] = 0
    df.loc[ (df.ADX > 0.2) , 'ADX_TREND'] = 1
    
    df['SQZ_ON'] = df.ta.squeeze().SQZ_ON
    
    df['RSI_FLAG'] = 0.5
    df.loc[(df.RSI_14 > 70), 'RSI_FLAG'] = 1
    df.loc[(df.RSI_14 < 30), 'RSI_FLAG'] = 0

    return df

# FYI: Rolling includes the current period.
def add_labels(df : pd.DataFrame()):
    value = 0.05
    df['is_buy'] = 0
    df['is_sell'] = 0
    df['BTO'] = 0

    max_high = df.high.rolling(4).max()
    min_low = df.low.rolling(4).min()
    #spread = max_high / min_low - 1
    #df['high_low_spread'] = spread / spread.rolling(20).mean()
    

    df['BTC'] = np.where(min_low / df.open - 1 < -value, 1, 0)
    df['STC'] = np.where(max_high / df.open - 1 > value, 1, 0)

    # Look forward
    lf_max_high = df.high.shift(3).rolling(4).max()
    lf_min_low = df.low.shift(3).rolling(4).min()

    df['BTO'] = np.where( lf_max_high / df.close - 1 > value, 1, 0)
    df['STO'] = np.where( lf_min_low / df.close - 1 < -value, 1, 0)
    df['target'] = np.where(df.BTO + df.STO == 0, 'hold', 'tbd')
    return None

def drop_crap(df: pd.DataFrame()):
    cols = df.columns
    for c in ['open','high', 'low', 'close', 'volume']:
        if c in cols:
            del df[c]
    return None

features = [ 
    'ADX',
    #'ADX_TREND', 
    'SQZ_ON',
            'RSI_9', 
            #'RSI_14', 'RSI_FLAG', 
            'PCTRET_1', 'PCTRET_5', 'PCTRET_10', 
            'SMA_9', 'SMA_21', 'SMA_34',
            'SMA_50', 
            #'SMA_100', 'SMA_150'
        ]


#x = df.loc[:, features].values
#y = df.loc[:,['target']].values

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def do_pca(df):
    x = df.loc[:, features].values
    y = df.loc[:, ['target']].values
    pca = PCA(n_components=2)

    x = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(x)
    print(f'PCA explained variance: {pca.explained_variance_ratio_}')

    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])
    principalDf.index = df.index
    finalDf = pd.concat([principalDf, df.target], axis=1)

    # Here we could apply the PCA comcept by reducing the number of features
    #features.append('pc1')
    #features.append('pc2')
    #df['pc1'] = finalDf.pc1
    #df['pc2'] = finalDf.pc2
    return

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['buy', 'sell', 'wipsaw']
    colors = ['g', 'r', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'],
                   finalDf.loc[indicesToKeep, 'pc2'], c=color, s=50)
    ax.legend(targets)
#do_pca(df)

# Round two
# test_size: what proportion of original data is used for test set
def model_train(df, features):
    train_img, test_img, train_lbl, test_lbl = train_test_split( df.loc[:, features].values, df.target.values, test_size=1/7.0, random_state=0)
    
    # Fit on training set only, but scale both training and test
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)
    
    # Make an instance of the Model
    pca = PCA(.99)
    pca.fit(train_img)
    #print(f'Componets required for desired variance: {pca.n_components_}')
    
    train_img = pca.transform(train_img)
    test_img = pca.transform(test_img)
    
    # Apply Logistic Regression to the Transformed Data
    # default solver is incredibly slow which is why it was changed to 'lbfgs'
    logisticRegr = LogisticRegression(solver = 'lbfgs')
    logisticRegr.fit(train_img, train_lbl)
    
    # Predict for One Observation (image)
    logisticRegr.predict(test_img[0].reshape(1,-1))
    
    # Predict for multiple Observation (image)
    logisticRegr.predict(test_img[0:10])
    
    # Accuracy
    score = logisticRegr.score(test_img, test_lbl)
    logisticRegr.__accuracy = score
    #print(f'Training Accuracy Score: {score:0.4%}')
    return pca, logisticRegr

def model_predict_one(pca, model, x):
    # Get x using df.iloc[-1][features].values
    x = pca.transform(x.reshape(1,-1))
    prediction = model.predict(x)[0]
    return prediction

# pd.Series(idxmax(s, 3), s.index[2:])
# idxmax(df.high, 4)
def idxmax(s, w :int()):
    i = 0
    size = len(s)
    nans = w - 1
    while i < nans:
        yield np.nan
        i += 1
    i = 0
    while i + w <= size:
        yield(s.iloc[i:i+w].idxmax())
        i += 1

def process_symbol(symbol):
    a0 = a1 = 0
    prices = pd.read_csv(f'datasets/{symbol}.csv.gz', index_col=0)
    df = add_features(prices)
    add_labels(df)
    #drop_crap(df)
    df.dropna(inplace=True)
    df['signal'] = df.target
    df = df[df.index > '2016-01-01'].copy()
    save_df = df.copy()
    
    df['target'] = np.where(df.BTO + df.STO == 0, 'hold', 'tbd')
    pca, model = model_train(df, features)
    last_row = save_df.iloc[-1][features].values
    expected = model_predict_one(pca, model, last_row)
    a0 = model.__accuracy
    #print(f'symbol={symbol:5s} prediction={expected} accuracy=')
    if expected == 'hold': return None
    
    # Train for buy/sell
    df = save_df.loc[(df.BTO == 1) | (df.STO == 1)].copy()   
    df.target = np.where(df.BTO == 1, 'buy', 'sell')
    
    pca, model = model_train(df, features)
    #last_rows = save_df.iloc[-1:2][features].values
    last_row = save_df.iloc[-1][features].values
    expected = model_predict_one(pca, model, last_row)
    a1 = model.__accuracy
    print(f'symbol={symbol:5s} prediction={expected:5s} a0={a0:0.2%} a1={a1:0.2%}')

##process_symbol('M')
#import sys
#sys.exit()

import re
import glob
p = re.compile('datasets/(.*?)\.')
for filename in glob.glob('datasets/*.csv.gz'):
    m = p.match(filename)
    symbol = m.group(1)
    process_symbol(symbol)

print('Done')
