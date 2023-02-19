        #!/usr/bin/python3
# # -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler

# %%
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

    df['SMA_STACKED'] = 0
    df.loc[(df.SMA_STACKED == 0) & (df.SMA_9 > df.SMA_21) & (df.SMA_21 > df.SMA_50) & (df.SMA_50 > df.SMA_100) , 'SMA_STACKED']  = -1
    df.loc[(df.SMA_STACKED == 0) & (df.SMA_9 < df.SMA_21) & (df.SMA_21 < df.SMA_50) & (df.SMA_50 < df.SMA_100) , 'SMA_STACKED']  = 1

    for i in [2, 14]:
        key = 'RSI_' + str(i)
        df[key] = df.ta.rsi(i) / 100
    df['RSI_SIGNAL'] = df.RSI_14 - df.RSI_14.rolling(14).mean();


    df['ADX'] = df.ta.adx().ADX_14/100
    df['ADX_TREND'] = 0
    df.loc[ (df.ADX > 0.2) , 'ADX_TREND'] = 1
    
    df['SQZ_ON'] = df.ta.squeeze().SQZ_ON.rolling(5).sum()/5      
    df['MACD'] = df.ta.macd().MACDs_12_26_9
    
    # https://usethinkscript.com/threads/trade-volume-delta-indicator-for-thinkorswim.524/
    df['CVD_BUYING'] = (df.close - df.low) / (df.high - df.low)
    df['CVD_SELLING'] = (df.high - df.close) / (df.high - df.low)
    df[['STOCHk_14_3_3', 'STOCHd_14_3_3']] = df.ta.stoch()

    return df

# FYI: Rolling includes the current period.
def add_labels(df: pd.DataFrame(), offset=5, atr_mult=0):
    df['ATR'] = df.ta.atr()
    value = 0.05
#    offset = 5

    # Look forward by 1st looking back, then walking forward
    lf_max_high = df.high.shift(-offset).rolling(offset).max()
    lf_min_low = df.low.shift(-offset).rolling(offset).min()

    df['BTO'] = np.where( lf_max_high / df.close - 1 > value, 1, 0)
    df['STO'] = np.where( lf_min_low / df.close - 1 < -value, 1, 0)
    if atr_mult > 0:
        df['BTO'] = np.where( lf_max_high > df.high + (df.ATR * atr_mult), 1, 0)
        df['STO'] = np.where( lf_min_low < df.low - (df.ATR * atr_mult), 1, 0)

    df['target'] = np.where(df.BTO + df.STO == 0, 'hold', 'tbd')
    return None

# %%
def prepare(symbol):
    df = pd.read_csv(f"datasets/{symbol}.csv.gz", index_col=0)
    df = add_features(df)
    add_labels(df)
    return df


def drop_crap(df: pd.DataFrame()):
    cols = df.columns
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in cols:
            del df[c]
    return None

features = [ 
    'ADX',
    'MACD',
    #'ADX_TREND',
    "SMA_STACKED",
    'SQZ_ON',
    'RSI_2', 'CVD_BUYING', 'CVD_SELLING', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    'RSI_14',
    'PCTRET_1', 'PCTRET_5', 'PCTRET_10', 
    'SMA_9', 'SMA_21', 'SMA_34',
    'SMA_50', 
    'SMA_100', 'SMA_150'
    ]


# %%
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def do_pca(df):
    x = df.loc[:, features].values
    y = df.loc[:, ['target']].values
    pca = PCA(n_components=2)

    x = StandardScaler().fit_transform(x)
    principalComponents = pca.fit_transform(x)
    #print(f'PCA explained variance: {pca.explained_variance_ratio_}')

    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])
    principalDf.index = df.index
    finalDf = pd.concat([principalDf, df.target], axis=1)
    finalDf['target'] = 'hold'
    finalDf.loc[ (df.BTO == 1 ) & (df.STO == 1), 'target' ] = 'wipsaw'
    finalDf.loc[ (df.BTO == 1 ) & (df.STO == 0), 'target' ] = 'buy'
    finalDf.loc[ (df.BTO == 0 ) & (df.STO == 1), 'target' ] = 'sell'

    # Here we could apply the PCA comcept by reducing the number of features
    #features.append('pc1')
    #features.append('pc2')
    #df['pc1'] = finalDf.pc1
    #df['pc2'] = finalDf.pc2
    return
'''
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = ['buy', 'sell'] #, 'wipsaw', 'hold']
    colors = ['limegreen', 'tab:red'] #, 'y', 'w']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'],
                   finalDf.loc[indicesToKeep, 'pc2'], c=color, s=50)
    ax.legend(targets)

    pca = PCA().fit(df.loc[:, features].values)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
'''
# %%
# Round two
# test_size: what proportion of original data is used for test set
def model_train(df, features):
    train_img, test_img, train_lbl, test_lbl = train_test_split( df.loc[:, features].values, df.target.values, test_size=0.25, random_state=0)
    
    # Fit on training set only, but scale both training and test
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)
    
    # Make an instance of the Model
    pca = PCA(.99)
    pca.fit(train_img)
    # print(f'Componets required for desired variance: {pca.n_components_}')
    
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
# %%
def tune_atr(symbol):
    for atr_mult in [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        for offset in [5, 10, 15]:
            df = pd.read_csv(f'datasets/{symbol}.csv.gz', index_col=0)
            df = add_features(df)
            add_labels(df, offset, atr_mult)
            df.dropna(inplace=True)
            df = df[df.index > '2016-01-01'].copy()
            save_df = df.copy()
        
            df['target'] = 'ICE'
            df.loc[ (df.BTO == 1 ) & (df.STO == 0), 'target' ] = 'BTO'
            df.loc[ (df.BTO == 0 ) & (df.STO == 1), 'target' ] = 'STO'
            pca0, model0 = model_train(df, features)
            expected00 = model_predict_one(pca0, model0, save_df.iloc[-1][features].values)
            a0 = model0.__accuracy
            print(f'symbol={symbol:5s} offset={offset:02d} atr_mult={atr_mult} score={a0:03.0%} p={expected00:3s}')
# offset=5, atr=0
# offset=5, atr=3.5

# %%
def process_symbol(symbol):
    atr_mult = 3.5
    a0 = a1 = a2 = 1
    df = pd.read_csv(f'datasets/{symbol}.csv.gz', index_col=0)
    df = add_features(df)
    add_labels(df, 5, atr_mult)
    df.dropna(inplace=True)
    #df['signal'] = df.target
    df = df[df.index > '2016-01-01'].copy()
    save_df = df.copy()

    ###########
    # One pass model
    df['target'] = 'ICE'
    df.loc[ (df.BTO == 1 ) & (df.STO == 0), 'target' ] = 'BTO'
    df.loc[ (df.BTO == 0 ) & (df.STO == 1), 'target' ] = 'STO'
    pca0, model0 = model_train(df, features)
    expected00 = model_predict_one(pca0, model0, save_df.iloc[-1][features].values)
    expected01 = model_predict_one(pca0, model0, save_df.iloc[-2][features].values)
    a0 = model0.__accuracy

    ##############
    # The Two Pass approach first trains a model on ICE/tbd
    df['target'] = np.where(df.BTO + df.STO == 0, 'ICE', 'tbd')
    pca1, model1 = model_train(df, features)
    expected10 = model_predict_one(pca1, model1, save_df.iloc[-1][features].values)
    expected11 = model_predict_one(pca1, model1, save_df.iloc[-2][features].values)
    a1 = model1.__accuracy
    
    # The 2nd pass is now left only to decide between buy/sell, but only if needed.
    if expected10 == expected11 == 'tbd':
        df = save_df.loc[(df.BTO == 1) | (df.STO == 1)].copy()
        df.target = np.where(df.BTO == 1, 'BTO', 'STO')
        pca2, model2 = model_train(df, features)
        expected10 = model_predict_one(pca2, model2, save_df.iloc[-1][features].values)
        expected11 = model_predict_one(pca2, model2, save_df.iloc[-2][features].values)
        a2 = model2.__accuracy

    # Count the matches (df.STO==1).sum()
    row = save_df.iloc[-1]
    score = (a0 + (a1 * a2))/2
    roi = row.ATR * atr_mult / row.close * score
    print(f'symbol={symbol:5s} score={score:03.0%} p={expected00:3s},{expected01:3s},{expected10:3s},{expected11:3s} a0={a0:03.0%},{a1:03.0%},{a2:03.0%} close={row.close:03.2f} atr={row.ATR:03.2f} roi={roi:06.2%}')

##process_symbol('M')
#import sys
#sys.exit()
#%%
def main():
    import re
    import glob
    p = re.compile('datasets/(.*?)\.')
    for filename in glob.glob('datasets/*.csv.gz'):
        filename = filename.replace('\\','/')
        m = p.match(filename)
        if m is None:
            print(f'filename={filename} NOMATCH')
            continue
        symbol = m.group(1)
        try:
            process_symbol(symbol)
        except:
            print(f'ERROR symbol={symbol}')
    
    print('Done')
#%%
# grep 'p0=STO p2=STO p1=STO' pca.out | sort -b -k 2 -r  | head -5
# grep 'p0=BTO p2=BTO p1=BTO' pca.out | sort -b -k 2 -r  | head -5
# symbol= 'ODFL'
# process_symbol(symbol)
if __name__ == "__main__":
    main()
