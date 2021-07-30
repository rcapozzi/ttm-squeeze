import sys
import time
import datetime as dt
import pandas as pd
import random
#import pandas_ta as ta
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import os
pd.options.mode.chained_assignment = None

# For the small ticker list of ETFs, these look optimal 35,45,160,0,12
def enumerate_params():
    keys = ['rsi_entry', 'rsi_exit', 'sma_period', 'stop_loss_pct', 'max_trade_days']
    params =  []
    i = 1
    for rsi_entry in range(15,36,5):
        for rsi_exit in range(20,46,5):
            if rsi_exit <= rsi_entry: continue
            for sma_period in range(100,201,25):
                for stop_loss_pct in [0, 5, 10]:
                    for max_trade_days in range(4,17,2):
                        param = dict(zip(keys, [rsi_entry, rsi_exit, sma_period, stop_loss_pct, max_trade_days
                ]))
                        i += 1
                        param['id'] = i
                        params.append(param)
    if OPTS['is_test']:
        params = random.sample(params,10)
        print(f'enumerate_params: test enabled params len={len(params)}')
    return params

def enumerate_params_products():
    keys = ['rsi_entry', 'rsi_exit', 'sma_period', 'stop_loss_pct', 'max_trade_days']
    params =  []
    r_rsi_entry = list(range(15,31,5))
    r_rsi_exit = list(range(35,46,5))
    r_sma_period = list(range(100,201,20))
    r_stop_loss_pct = list(range(0,9,2))
    r_max_trade_days = list(range(4,17,2))
    list(itertools.product(r_rsi_entry,r_rsi_exit, r_sma_period, r_stop_loss_pct, r_max_trade_days))
    return params

def in_trade(idx, trades):
    for t in trades:
        if t[1] <= idx and t[3] > idx: return True
    return False

def rsi_momo_strategy(symbol, df, params):
    #print(f'rsi_momo_strategy: {symbol} {params}')
    stop_loss_pct = params['stop_loss_pct']
    df = df.copy()
    #df['rsi'] = df.ta.rsi()
    #df['sma'] = df.ta.sma(params['sma_period'])
    df['rsi'] = ta.momentum.rsi(df.close,window=10)
    df['sma'] = ta.trend.sma_indicator(df.close, window=params['sma_period'])
    df['atr'] = ta.volatility.AverageTrueRange(df.high, df.low, df.close).average_true_range() 
    df['roc'] = ta.momentum.ROCIndicator(df.close).roc() 
    #df['adx'] = ta.trend.ADXIndicator(df.high, df.low, df.close, 14, True).adx() 
    df['adx'] = 0

    df.dropna(inplace=True)
    
    # The DF is daily, and therefor has no weekends and timedelta is not needed.
    cutoff = df.tail(params['max_trade_days']+1).index.min()

    all_trades = []
    def apply_trade(row, tag):
        if in_trade(row.name, trades): return None
        iloc = df.index.get_loc(row.name)
        buy_day = df.iloc[iloc+1]
        sell_stop = None
        if params['stop_loss_pct']:
            sell_stop = buy_day.open * (1.0 - (params['stop_loss_pct']/100))
        sell_day = None
        sell_descr = 'max_days'
        for j in range(1,params['max_trade_days']):
            this_day = df.iloc[iloc + j]
            sell_day = df.iloc[iloc + j + 1]
            if this_day.rsi > params['rsi_exit']:
                sell_descr = 'rsi_exit'
                break
            if sell_stop and this_day.low < sell_stop:
                sell_descr = 'stop_loss'
                break
        pct_return = sell_day.open / buy_day.open - 1
        days_open = j
        data = [symbol, buy_day.name, buy_day.open, sell_day.name, sell_day.open, pct_return, sell_descr, \
                       params['id'], j, tag, buy_day.rsi, buy_day.sma, buy_day.atr, buy_day.roc, buy_day.adx]
        trades.append(data)
        all_trades.append(data)

    # RSI xUnder
    trades = []
    signals = df.loc[(df.close > df.sma) & (df.rsi < params['rsi_entry']) & (df.shift(1).rsi < params['rsi_entry']) & (df.index < cutoff)]
    signals.apply(lambda row: apply_trade(row, 'rsi_xunder'), axis=1)

    # RSI xOver
    trades = []
    signals = df.loc[(df.close > df.sma) & (df.rsi > params['rsi_entry']) & (df.shift(1).rsi < params['rsi_entry']) & (df.index < cutoff)]
    signals.apply(lambda row: apply_trade(row, 'rsi_xover'), axis=1)

    df = pd.DataFrame(all_trades, columns=['symbol', 'bdate', 'bprice', 'sdate', 'sprice', 'pct_return', 'sell_descr', \
        'param_id', 'days_open', 'strategy', 'rsi', 'sma', 'atr', 'roc', 'adx'])
    return df

def myRSI(symbol):
    df['Upmove'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
    df['avg_down'] = df.Downmove.ewm(span=19).mean()

def flatten(t):
    return [item for sublist in t for item in sublist]

def load_symbol(symbol):
    file = f"datasets/{symbol}.csv.gz"
    if os.path.isfile(file):
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
    else:
        period = '3mo'
        df = yf.download(symbol, progress=False, start='2005-01-01')
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        df.set_index('date', inplace=True)
        df.symbol = symbol       
        df.to_csv(file,index=True)
    df['symbol'] = symbol
    df.name = symbol
    if OPTS['df_start_on']: df = df[df.index >= OPTS['df_start_on']]
    if OPTS['df_end_on']: df = df[df.index <= OPTS['df_end_on']]
    return df


def uberdf_load_all(filename):
    symbols = []
    start = time.time()
    frames = {}

    if type(filename) == str:
        with open(filename) as f:
            lines = f.read().splitlines()
            for s in lines: symbols.append(s)
    if type(filename) == list: symbols = filename

    # for filename in os.listdir('datasets'):
    #     s = filename.split(".")[0]
    #     symbols.append(s)

    for symbol in symbols:
        df = load_symbol(symbol)
        if df.empty: continue
        frames[symbol] = df
    end = time.time()
    print(f'uberdf_load_all: elapsed: {end - start:0.2f} size={len(frames)}')
    return frames

def uberdf_one_config(udf, params):
    print(f'uberdf_one_config: params={params}')
    start = time.time()    
    frames = []
    for symbol, df in udf.items():
        df = rsi_momo_strategy(symbol, df, params)
        if len(df) > 0:
            frames.append(df)
    if len(frames) == 0:
        return

    df = pd.concat(frames)
    for k, v in params.items():
        df[k] = v                            
    filename=f'results/trades.{params["id"]}.csv.gz'
    print(f'uberdf_one_config: write df={filename}')
    df.to_csv(filename,index=False)

    params['trades'] = len(df)
    print(f'uberdf_one_config: results={params}')
    return df

def uberdf_shard(shard_id=None,shard_max=None):
    print(f'uberdf_shard: id={shard_id} shard_max={shard_max}')
    start = time.time()
    configs = enumerate_params()
    for params in configs:
        i = params['id']
        if i % shard_max != shard_id: continue
        uberdf_one_config(udf, params)
    end = time.time()
    print(f'uberdf_shard: Elapsed: {end - start:0.2f}')
    return configs

OPTS = {
    'is_test': False,
    'df_start_on': '2007-01-01',
    'df_end_on': '2021-06-01',
}
XOPTS = {
    'is_test': True,
    'df_start_on': '2016-01-01',
    'df_end_on': '2021-01-01',
}

if __name__ == "__main__":
    symbol_file = sys.argv[1]
    shard_id = int(sys.argv[2]) - 1
    shard_max = int(sys.argv[3])

    udf = uberdf_load_all(symbol_file)
    results = uberdf_shard(shard_id, shard_max)
