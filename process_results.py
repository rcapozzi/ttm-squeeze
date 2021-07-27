#!/usr/bin/env python

import os
import sys

keys = ['rsi_entry',
 'rsi_exit',
 'sma_period',
 'stop_loss_pct',
 'max_lookahead',
 'id',
 'elapsed',
 'trades',
 'wins',
 'pct_wins',
 'mean',
 'std',
 'sum']

def process_results_file(file):
    lines = None
    with open(file, 'r') as file:
        lines = file.read().splitlines()

    for line in lines:
        data = eval(line)
        values = []
        for k in keys: values.append(data[k])
        v = [ str(x) for x in values ]
        v = ','.join(v)
        print(v)

v = ','.join(keys)
print("#" + v)
for f in sys.argv[1:]: process_results_file(f)
