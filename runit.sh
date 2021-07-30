#!/bin/sh

#exec python3 ./rsi_momo.py $1 $2 > results/rsi_momo.$1.out

#mkdir -p results
#seq 1 10 | xargs -l -I'{}' -P 3 runone {} 10

# seq 1 64 | xargs -I X -P 8 sh -c 'python3 ./rsi_momo.py $1 64 > results/rsi_momo.$1.out' -- X

rm results/*.csv.gz results/*.out
seq 1 32 | xargs -I X -P 6 sh -c 'python3 ./rsi_momo.py ./sector-etf.txt $1 32 > results/rsi_momo.$1.out' -- X
python3 ./process_results.py