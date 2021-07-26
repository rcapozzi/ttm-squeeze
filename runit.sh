#!/bin/sh

runone(){
	python3 ./rsi_momo.py $1 $2 > rsi_momo.$1.out
}

seq 1 10 | xargs -l -I'{}' -P 4 ./runit.sh {} 10
