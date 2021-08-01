#!/usr/bin/env python
import os
import sys
import re
import pandas as pd

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

import glob
def process_trades():
    ary = []
    for filename in glob.glob('results/*.csv.gz'):
        df = pd.read_csv(filename)
        if len(df) == 0: continue
        ary.append(df)
    df = pd.concat(ary)
    df['pct_sma'] = df.sma / df.bprice - 1
    df['pct_atr'] = df.atr / df.bprice
    df.to_csv('trades.csv',index=False)

#     with pd.ExcelWriter('trades.xlsx') as writer:
#         df.to_excel(writer, sheet_name='raw')
    #df.to_excel('trades.xlsx')
    sample = df.sample(300_000)
    sample.to_excel('trades.xlsx')
    # df[(df.bdate >= '2021-04-01') & (df.rsi_entry == 15) & (df.rsi_exit == 45) & (df.stop_loss_pct == 10) & (df.sma_period == 150) & (df.max_trade_days == 8)]
    return df

df = process_trades()

#v = ','.join(keys)
#print("#" + v)
#for f in sys.argv[1:]: process_results_file(f)
