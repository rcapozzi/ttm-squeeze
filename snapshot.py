import os
import yfinance as yf

with open('symbols.csv') as f:
    lines = f.read().splitlines()
    for symbol in lines:
        file = "datasets/{}.csv.gz".format(symbol)
        if os.path.isfile(file):
            print('INFO: skilling {symbol}')
            continue
        print('INFO: downloading {symbol}')
        data = yf.download(symbol, start="2005-01-01") #, end="2020-08-22")
        data.to_csv(file)
