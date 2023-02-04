# ttm-squeeze
/Users/rcapozzi/Library/Python/3.9/bin/jupyter-lab

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

# TODO
pycaret see hackernoon

# Thinkscript
Based on https://tosindicators.com/indicators/automated-trading

(1) is previous bar. (2) is two bars un the past.
Sell put plot signal = RSI().DownSignal(1);

Buy when if EMA8/21/34 stacked and close < EMA34
Scenario 7: Buy call if squeeze, greater than avg vol, and emas stacked.
def ema8 = ExpAverage(close, 8);
def ema21 = ExpAverage(close, 21);
def ema34 = ExpAverage(close, 34);
def bullish = if ema8 > ema21 and ema21 > ema34 then 1 else 0;
def squeeze = if TTM_Squeeze().SqueezeAlert == 0 then 1 else 0;
def greaterVol = VolumneAvg().Vol > VolumeAvg().VolAvg and close > close[1];
def conditions = bullish and squeeze and greaterVol;
plot signal = conditions[1];

Scenario 8a:
def ema8 = ExpAverage(close, 8);
def ema21 = ExpAverage(close, 21);
def ema34 = ExpAverage(close, 34);
def stacked = if ema8 > ema21 and ema21 > ema34 then 1 else 0;

def IV = IMP_VOLATILITY();
def IV8 = ExpAverage(IV, 8);
def IV21 = ExpAverage(IV, 21);
def IV34 = ExpAverage(IV, 34);
def IVstacked = if IV8 > IV21 and IV21 > IV34 then 1 else 0;
def cond = if stacked and IVstacked and IV < EMA34 and close > close[1] then 1 else 0;
plot signal = conditions[1];
Exit when ema no longer stacked 
# addOrder(OrderType.BUY_AUTO, no);

```
#
# TD Ameritrade IP Company, Inc. (c) 2013-2021
#

input price = hlc3;
input averageLength = 8;
input volatilityLength = 13;
input deviationFactor = 3.55;
input lowBandAdjust = 0.9;

def typical = if price >= price[1] then price - low[1] else price[1] - low;
def deviation = deviationFactor * Average(typical, volatilityLength);
def devHigh = ExpAverage(deviation, averageLength);
def devLow = lowBandAdjust * devHigh;
def medianAvg = ExpAverage(price, averageLength);

plot MidLine = Average(medianAvg, averageLength);
plot UpperBand = ExpAverage(medianAvg, averageLength) + devHigh;
plot LowerBand = ExpAverage(medianAvg, averageLength) - devLow;

LowerBand.SetDefaultColor(GetColor(1));
MidLine.SetDefaultColor(GetColor(7));
UpperBand.SetDefaultColor(GetColor(0));

AddOrder(OrderType.BUY_AUTO, close > UpperBand, name = "VolatilityBandLE", tickcolor = GetColor(0), arrowcolor = GetColor(0));
AddOrder(OrderType.SELL_AUTO, close < LowerBand, name = "VolatilityBandSE", tickcolor = GetColor(1), arrowcolor = GetColor(1));

#
# TD Ameritrade IP Company, Inc. (c) 2012-2021
#

input price = FundamentalType.LOW;
input smaLength = 50;
input trendLength = 11;

def priceVix = Fundamental(price, "VIX");
def smaVix = Average(priceVix, smaLength);

AddOrder(OrderType.BUY_AUTO,
    Sum(priceVix < smaVix, trendLength)[1] == trendLength,
    tickcolor = GetColor(1),
    arrowcolor = GetColor(1),
    name = "VIX_Timing_LE");
AddOrder(OrderType.SELL_TO_CLOSE,
    Sum(priceVix > smaVix, trendLength)[1] == trendLength,
    tickcolor = GetColor(2),
    arrowcolor = GetColor(2),
    name = "VIX_Timing_LX");
```

```
# IVR
def ivol = if!isNaN(imp_volatility) then imp_volatility else ivol;
def lowvol = lowest(ivol,60);
def highvol = highest(ivol,60);
def currentvol = imp_volatility;
plot data = ((currentvol - lowvol)/(highvol - lowvol)*100);
```

```
def jan = getMonth() == 1;
AddChartBubble(jan, close, jan, color.yellow);
```

ODFL: 2021-07-22 lf-5-day-high is 7%. should be BTO
