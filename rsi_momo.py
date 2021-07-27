import sys
import time
import datetime as dt
import pandas as pd
#import pandas_ta as ta
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import os
pd.options.mode.chained_assignment = None

def enumerate_params():
    keys = ['rsi_entry', 'rsi_exit', 'sma_period', 'stop_loss_pct', 'max_trade_days']
    params =  []
    i = 1
    for rsi_entry in range(15,36,5):
        for rsi_exit in range(20,46,5):
            if rsi_exit <= rsi_entry: continue
            for sma_period in range(100,201,20):
                for stop_loss_pct in range(0,6,1):
                    for max_trade_days in range(4,17,2):
                        param = dict(zip(keys, [rsi_entry, rsi_exit, sma_period, stop_loss_pct, max_trade_days
                ]))
                        param['id'] = i
                        params.append(param)
    return params

def enumerate_params_products():
    keys = ['rsi_entry', 'rsi_exit', 'sma_period', 'stop_loss_pct', 'max_trade_days']
    params =  []
    r_rsi_entry = list(range(15,31,5))
    r_rsi_exit = list(range(35,46,5))
    r_sma_period = list(range(100,201,20))
    r_stop_loss_pct = list(range(0,6,1))
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

    df.dropna(inplace=True)
    
    # The DF is daily, and therefor has no weekends and timedelta is not needed.
    cutoff = df.tail(params['max_trade_days']+1).index.min()
    # A Cross under from prior period
    #signals = df.loc[(df.close > df.sma) & (df.rsi < params['rsi_entry']) & (df.index < cutoff)]
    # A Cross over from prior period
    signals = df.loc[(df.close > df.sma) & (df.rsi > params['rsi_entry']) & (df.shift(1).rsi < params['rsi_entry']) & (df.index < cutoff)]
    trades = []
    for idx, row in signals.iterrows():
        if in_trade(idx, trades): continue
        iloc = df.index.get_loc(idx)
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
        #print(f'i:{i:03d} iloc:{iloc:05d} buy_on:{buy_day.name} sell_on:{sell_day.name}')
        trades.append([symbol, buy_day.name, buy_day.open, sell_day.name, sell_day.open, pct_return, sell_descr])
    df = pd.DataFrame(trades, columns=['symbol', 'bdate', 'bprice', 'sdate', 'sprice', 'pct_return', 'sell_descr'])
    df['strategy'] = 'rsi_xover'
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
        df.columns = df.columns.str.lower()
        df.to_csv(file,index=True)
    df['symbol'] = symbol
    df.name = symbol
    if OPTS and OPTS['df_start_on']: df = df[df.index >= OPTS['df_start_on']]
    if OPTS and OPTS['df_end_on']: df = df[df.index <= OPTS['df_end_on']]
    return df


def uberdf_load_all(filename):
    symbols = []
    start = time.time()
    frames = {}

    with open(filename) as f:
        lines = f.read().splitlines()
        for s in lines: symbols.append(s)

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
    for symbol in udf.keys():
        df = udf[symbol]
        df = rsi_momo_strategy(symbol, df, params)
        frames.append(df)

    df = pd.concat(frames)
    end = time.time()
    params['elapsed'] = end - start
    params['mean'] = 0
    params['std'] = 0
    params['sum'] = 0

    params['trades'] = len(df)
    if len(df) > 0:        
        params['wins'] = wins = df.loc[df.pct_return > 0].pct_return.count()
        params['pct_wins'] = params['wins'] / len(df)
    else:
        params['wins'] = 0
        params['pct_wins'] = 0
        params['mean'] = df.pct_return.mean() 
        params['std'] = df.pct_return.std()
        params['sum'] = df.pct_return.sum()
    print(f'uberdf_one_config: results={params}')
    #print(f'uberdf_one_config: mean={params["mean"]:04.2f} std={params["sum"]:04.2f} sum={params["std"]:04.2f}')
    #print(f'Summary: wins={wins} total={total} rate={wins/total*100:04.2f} pct_return={df.pct_return.mean()*100:05.2f}%')
    #print(df.pct_return.describe())
    return df

def uberdf_main(shard_id=None,shard_max=None):
    print(f'uberdf_main: id={shard_id} shard_max={shard_max}')
    start = time.time()
    configs = enumerate_params()
    for params in configs:
        i = params['id']
        if i % shard_max != shard_id: continue
        uberdf_one_config(udf, params)
    end = time.time()
    print(f'uberdf_main: Elapsed: {end - start:0.2f}')
    return configs

# Process command line
symbol_file = sys.argv[1]
shard_id = int(sys.argv[2])
shard_max = int(sys.argv[3])

OPTS = {
    'df_start_on': '2007-01-01',
    'df_end_on': '2013-01-01',
}
udf = uberdf_load_all(symbol_file)
results = uberdf_main(shard_id, shard_max)

