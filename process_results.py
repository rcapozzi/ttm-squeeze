#!/usr/bin/env python3
import os
import pandas as pd
import re
import sys
import ta
import glob

keys = ['rsi_entry',
 'rsi_exit',
 'sma_period',
 'stop_loss_pct',
 'max_trade_days',
 'id',
 'elapsed',
 'trades',
 'wins',
 'pct_wins',
 'mean',
 'std',
 'sum']


def process_results_file(file):
    p = re.compile('(uberdf_one_config: results=)(.*)')
    lines = None
    with open(file, 'r') as file:
        lines = file.read().splitlines()

    for line in lines:
        m = p.match(line)
        if not m: continue
        text = m.group(2).replace('nan','0')
        data = eval(text)
        values = []
        for k in keys: values.append(data[k])
        v = [ str(x) for x in values ]
        v = ','.join(v)
        print(v)

def enrich_add_ta(df):
    df['rsi10'] = ta.momentum.rsi(df.close,window=10)
    df['rsi14'] = ta.momentum.rsi(df.close,window=14)
    for i in [10, 20, 50, 100, 200]:
        s = ta.trend.sma_indicator(df.close, window=i)
        df['pct_sma'+str(i)] = s / df.close - 1
    df['pct_atr'] = ta.volatility.AverageTrueRange(df.high, df.low, df.close).average_true_range() / df.close 
    df['roc'] = ta.momentum.ROCIndicator(df.close).roc() 
    df['adx'] = ta.trend.ADXIndicator(df.high, df.low, df.close, 14, True).adx() 
        #df['adx'] = 0

def enrich_get_symbol(symbol,symbols):
    if symbol in symbols:
        return symbols[symbol]
    # Gotta load it
    filename = f'datasets/{symbol}.csv.gz'
    print(f'INFO: enrich_get_symbol current_size={len(symbols)} loading={filename}')
    df = pd.read_csv(filename, index_col='date', parse_dates=True)
    enrich_add_ta(df)
    symbols[symbol] = df
    return df

def enrich_df_i(row,symbols):
    data = enrich_get_symbol(row.symbol, symbols)
    data = data.loc[row.bdate]
    # We don't want to call data.columns.to_list() for every trade
    for i in ['rsi10', 'rsi14', 'pct_sma10', 'pct_sma20', 'pct_sma50', 'pct_sma100', 'pct_sma200', 'pct_atr', 'roc', 'adx']:
        row[i] = data[i]
    return None

def enrich_df(df):
    df.apply(lambda row: enrich_df_i(row, symbols), axis=1)
    return df

def process_trades():
    CSV_APPEND = False
    ary = []
    for filename in glob.glob('results/*.csv.gz'):
        df = pd.read_csv(filename)
        if len(df) == 0: continue
        enrich_df(df)
        if CSV_APPEND:
            df.to_csv('trades.csv', index=False, mode='a')    
        else:
            ary.append(df)
        print(f'INFO: processed={filename}')

    print(f'INFO: running concat')
    df = pd.concat(ary)
    print(f'INFO: writing trades.csv')
    df.to_csv('trades.csv',index=False)

#     with pd.ExcelWriter('trades.xlsx') as writer:
#         df.to_excel(writer, sheet_name='raw')
    #df.to_excel('trades.xlsx')
    sample = df.sample(300_000)
    sample.to_excel('trades.xlsx')
    # df[(df.bdate >= '2021-04-01') & (df.rsi_entry == 15) & (df.rsi_exit == 45) & (df.stop_loss_pct == 10) & (df.sma_period == 150) & (df.max_trade_days == 8)]
    return df

if __name__ == "__main__":
    symbols = {}
    symbol_file = sys.argv[1]
    df = process_trades()

#v = ','.join(keys)
#print("#" + v)
#for f in sys.argv[1:]: process_results_file(f)
