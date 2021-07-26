#!/bin/sh

exec python3 ./rsi_momo.py $1 $2 > results/rsi_momo.$1.out

#mkdir -p results
#seq 1 10 | xargs -l -I'{}' -P 3 runone {} 10
