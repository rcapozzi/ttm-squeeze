#!/bin/sh

# mkdir -p results
# seq 1 64 | xargs -I X -P 8 sh -c 'python3 ./rsi_momo.py $1 64 > results/rsi_momo.$1.out' -- X

rm -fr results
mkdir -p results

seq 1 64 | xargs -I X -P 6 sh -c 'python3 ./rsi_momo.py ./sp500.csv $1 64 > results/rsi_momo.$1.out' -- X
python3 ./process_results.py
