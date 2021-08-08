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

# prices['log_ret'] = np.log(prices.close / prices.close.shift(1))

def ema_stacked(df):
    df['SMA_STACKED_BULL'] = (df.SMA_8 >= df.SMA_16) & (df.SMA_16 >= df.SMA_32) & (df.SMA_32 > df.SMA_64)
    df['SMA_STACKED_BEAR'] = (df.SMA_8 <= df.SMA_16) & (df.SMA_16 <= df.SMA_32) & (df.SMA_32 <= df.SMA_64)

def add_features(df :pd.DataFrame):
    features = []
    prices.ta.percent_return(1,append=True)
    prices.ta.percent_return(5,append=True)
    prices.ta.percent_return(10,append=True)

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

def add_labels(df : pd.DataFrame()):
    value = 0.05
    rets = []
    df['is_buy'] = 0
    df['is_sell'] = 0

    for i in range(4):
        #df['ret'+i] = df.shift(-i).high / df.close - 1
        df.loc[(df['is_buy'] == 0) & (df.shift(-i).high / df.close - 1 > value), 'is_buy'] = 1
        df.loc[(df['is_sell'] == 0) & (df.shift(-i).low / df.close - 1 < -value), 'is_sell'] = 1

    df['target'] = 'X'
    df.loc[(df.target == 'X') & (df.is_buy == 1) & (df.is_sell == 1) , 'target'] = 'hold'
    df.loc[(df.target == 'X') & (df.is_buy == 1), 'target'] = 'buy'
    df.loc[(df.target == 'X') & (df.is_sell == 1), 'target'] = 'sell'
    df.loc[(df.target == 'X'), 'target'] = 'hold'
    del df['is_buy']
    del df['is_sell']
    return None

def drop_crap(df: pd.DataFrame()):
    cols = df.columns
    for c in ['open','high', 'low', 'close', 'volume']:
        if c in cols:
            del df[c]
    return None

features = [ 'ADX', 'ADX_TREND', 'SQZ_ON',
            'RSI_9', 'RSI_14', 'RSI_FLAG', 
            'PCTRET_1', 'PCTRET_5', 'PCTRET_10', 
            'SMA_9', 'SMA_21', 'SMA_34', 'SMA_50',  'SMA_100', 'SMA_150'
        ]

prices = pd.read_csv('datasets/AAPL.csv.gz', index_col=0)

df = add_features(prices)
add_labels(df)
drop_crap(df)
df.dropna(inplace=True)
df = df[df.index > '2016-01-01'].copy()

x = df.loc[:, features].values
y = df.loc[:,['target']].values

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
principalDf.index = df.index
finalDf = pd.concat([principalDf, df.target], axis = 1)

# Here we could apply the PCA comcept by reducing the number of features
features.append('pc1')
features.append('pc2')
df['pc1'] = finalDf.pc1
df['pc2'] = finalDf.pc2


import matplotlib.pyplot as plt
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['buy', 'sell', 'hold']
colors = ['g', 'r', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
               , finalDf.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
pca.explained_variance_ratio_
    
# Round two

from sklearn.model_selection import train_test_split
# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( df.loc[:, features].values, df.target.values, test_size=1/7.0, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95)
pca.fit(train_img)
# Componets required for 95% of variance
pca.n_components_
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# Apply Logistic Regression to the Transformed Data
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_img, train_lbl)
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))
# Predict for multiple Observation (image)
logisticRegr.predict(test_img[0:10])

# Accuracy
print(f'Accuracy: {logisticRegr.score(test_img, test_lbl)}')
