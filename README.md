# ttm-squeeze
TTM Squeeze Scanner For Stocks in Python using Pandas and YFinance

### Tutorial Video / Screencast on how to build this scanner can be found on my YouTube channel

https://www.youtube.com/watch?v=YhkNoOqYp9A

snapshot.py
* Make this multi-threads. The YF download is multi-threaded, but you'd need to batch the symbols and you need to figure out the min start for that batch.

# PCA
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html


# All about calculating returns

For daily, log returns, use df['log_returns'].cumsum().apply(np.exp)

(1+pct_change).cumprod() - 1

```(1+df.value).rolling(window=X).agg(np.prod) - 1```
Faster
```(1 + df.value).rolling(window=X).apply(np.prod, raw=True) - 1```

The two series have the same total return:
1, 1, 1, 1, 2
1, 2, 3, 4, 2

# Data sources
https://www.alphavantage.co/#about
https://rapidapi.com/twelvedata/api/twelve-data1/details
https://iexcloud.io/cloud-cache/

# 
backtrader https://algotrading101.com/learn/backtrader-for-backtesting/
zipline

# Tear sheets
 quantstats handles a single series of returns
 Pyfolio Reloaded https://pyfolio.ml4trading.io/index.html
