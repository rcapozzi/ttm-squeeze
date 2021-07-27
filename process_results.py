#!/usr/bin/env python
import os
import sys
import re

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

v = ','.join(keys)
print("#" + v)
for f in sys.argv[1:]: process_results_file(f)
