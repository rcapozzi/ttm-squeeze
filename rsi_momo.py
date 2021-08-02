import sys
import time
import datetime as dt
import numpy as np
import pandas as pd
#import pandas_ta as ta
import random
import ta
import yfinance as yf
import matplotlib.pyplot as plt
import os
pd.options.mode.chained_assignment = None

def rsi_momo_strategy(symbol, df, params):
    #print(f'rsi_momo_strategy: {symbol} {params}')
    if 'trade_overlap' not in params:
        params['trade_overlap'] = 0
    all_trades = []
    last_trade_dt = None
    # The DF is daily, and therefor has no weekends and timedelta is not needed.
    # cutoff = df.tail(params['max_trade_days']+1).index.min()
    cutoff = df.iloc[-25].name

    def setup(rsi_period):
        nonlocal last_trade_dt
        last_trade_dt = None
        df['rsi'] = df['rsi' + str(rsi_period)]
        df['sma'] = df['sma' + str(params['sma_period'])]
        return df

    def apply_trade(row, tag):
        nonlocal last_trade_dt
        overlap = 0
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
        if (last_trade_dt and buy_day.name <= last_trade_dt):
            #print(f'd0={buy_day.name} d1={last_trade_dt} ?={buy_day.name <= last_trade_dt} overlap={overlap}')
            params['trade_overlap'] += 1
            overlap += 1
        last_trade_dt = sell_day.name
        pct_return = sell_day.open / buy_day.open - 1
        days_open = j
        data = [symbol, buy_day.name, buy_day.open, sell_day.name, sell_day.open, pct_return, sell_descr, overlap, \
                       params['id'], j, tag]
        all_trades.append(data)

    # RSI xUnder
    df = setup(10)
    signals = df.loc[(df.close > df.sma) & (df.rsi < params['rsi_entry']) & (df.shift(1).rsi > params['rsi_entry']) & (df.index < cutoff)]
    signals.apply(lambda row: apply_trade(row, 'rsi_xunder'), axis=1)

    # RSI xOver
    df = setup(14)
    signals = df.loc[(df.close > df.sma) & (df.rsi > params['rsi_entry']) & (df.shift(1).rsi < params['rsi_entry']) & (df.index < cutoff)]
    signals.apply(lambda row: apply_trade(row, 'rsi_xover'), axis=1)

    df = pd.DataFrame(all_trades, columns=['symbol', 'bdate', 'bprice', 'sdate', 'sprice', 'pct_return', 'sell_descr', 'overlap', \
            'param_id', 'days_open', 'strategy'])
    return df

# For the small ticker list of ETFs, these look optimal 35,45,160,0,12
def enumerate_params():
    keys = ['rsi_entry', 'rsi_exit', 'sma_period', 'stop_loss_pct', 'max_trade_days']
    params =  []
    i = 1
    for rsi_entry in range(15,36,5):
        for rsi_exit in range(20,46,5):
            if rsi_exit <= rsi_entry: continue
            for sma_period in range(50,201,50):
                for stop_loss_pct in [10]: # , 5, 10]:
                    for max_trade_days in [2, 4, 8, 16]:
                        param = dict(zip(keys, [rsi_entry, rsi_exit, sma_period, stop_loss_pct, max_trade_days ]))
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

def flatten(t):
    return [item for sublist in t for item in sublist]

def uberdf_enrich(udf, params):
    sma_periods = np.unique([h['sma_period'] for h in params])
    
    for symbol, df in udf.items():
        df['rsi10'] = ta.momentum.rsi(df.close,window=10)
        df['rsi14'] = ta.momentum.rsi(df.close,window=14)
        for i in sma_periods:
            df['sma'+str(i)] = ta.trend.sma_indicator(df.close, window=i)
        #df['sma20'] = ta.trend.sma_indicator(df.close, window=20)
        #df['atr'] = ta.volatility.AverageTrueRange(df.high, df.low, df.close).average_true_range() 
        #df['roc'] = ta.momentum.ROCIndicator(df.close).roc() 
        #df['adx'] = ta.trend.ADXIndicator(df.high, df.low, df.close, 14, True).adx() 
        #df['adx'] = 0
        df.dropna(inplace=True)

def yf_df_normalize(symbol, df):
    #if 'date' in df: 
    #    df.drop('date',axis=1, inplace=True)
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()

    if df.index.dtype == np.int64 and df.date.dtype == 'O':
        df.date = pd.to_datetime(df.date).dt.normalize()
    df.set_index('date', inplace=True)
    if 'adj close' in df: 
        df['close'] = df['adj close']
        df.drop('adj close',axis=1, inplace=True)
    df.symbol = symbol
    return df

# Return empty dataframe when symbol file is not found
def load_symbol(symbol):
    file = f"datasets/{symbol}.csv.gz"
    if not os.path.isfile(file):
        return pd.DataFrame()
        df = yf.download(symbol, progress=False, start='2005-01-01')
        df = yf_df_normalize(symbol, df)
        df.to_csv(file,index=True)

    if os.path.isfile(file):
        #df = pd.read_csv(file, index_col=0, parse_dates=True)
        df = pd.read_csv(file)
        df = yf_df_normalize(symbol, df)

    df.symbol = symbol
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
    #print(f'uberdf_one_config: params={params}')
    start = time.time()    
    frames = []
    for symbol, df in udf.items():
        df = rsi_momo_strategy(symbol, df, params)
        if len(df) > 0:
            frames.append(df)
    params['trades'] = 0
    if len(frames) > 0:
        df = pd.concat(frames)
        for k, v in params.items():
            df[k] = v                            
        filename=f'results/trades.{params["id"]:04d}.csv.gz'
        df.to_csv(filename,index=False)
        params['results'] = filename
        params['trades'] = len(df)

    print(f'uberdf_one_config: results={params}')
    return df

def uberdf_shard(shard_id=None,shard_max=None):
    print(f'uberdf_shard: id={shard_id} shard_max={shard_max}')
    start = time.time()
    configs = enumerate_params()
    uberdf_enrich(udf, configs)
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
    'df_end_on': '2021-07-01',
}
OPTS = {
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
