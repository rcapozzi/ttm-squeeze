#!/bin/sh
python ./snapshot.py
python ./pca.py > pca.out

for c in BTO,ICE STO,ICE; do grep p=$c pca.out | sort -b -k 7 -r  | head -2; grep p=$c pca.out | sort -b -k 2 -r  | head -2; done
