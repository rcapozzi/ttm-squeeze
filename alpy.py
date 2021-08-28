APCA_API_BASE_URL="https://paper-api.alpaca.markets"
APCA_API_KEY_ID="PK1WXIMQBGHTKTOFYP6L"
APCA_API_SECRET_KEY="kIjRsAyv9YDzZl4tV0rf7NTb224s0ICpw5I9CRpc"

from config import *
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

import requests, json
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

headers = {
 'APCA-API-KEY-ID': APCA_API_KEY_ID,
'APCA-API-SECRET-KEY': APCA_API_SECRET_KEY
}

APCA_API_ACCOUNT_URL=APCA_API_BASE_URL + "/v2/account"
APCA_API_ORDERS_URL=APCA_API_BASE_URL + "/v2/orders"

#%%
def get_account():
    r = requests.get(APCA_API_ACCOUNT_URL, headers=headers)
    return json.loads(r.content)

#%%
def get_orders():
    r = requests.get(APCA_API_ORDERS_URL, headers=headers)
    return json.loads(r.content)
orders = get_orders()    

# %%
def create_order(symbol, qty, side):
    type = "market"
    time_in_force = "gtc"
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force,
    }
    
    r = requests.post(APCA_API_ORDERS_URL, headers=headers, json=data)
    return json.loads(r.content)

def short_order(symbol, qty):
    type = "limit"
    time_in_force = "day"
    limit_price = 390.00
    
    data = {
         "symbol": symbol,
         "qty": qty,
         "side": 'sell',
         "type": type,
         'limit_price': limit_price,
         "time_in_force": time_in_force,
         "order_class": "bracket",
         'take_profit': {
                 'limit_price': limit_price * 0.95,
                 "time_in_force": "gtc",
             },
         'stop_loss': {
            'stop_price': limit_price * 1.050,
            'limit_price': limit_price * 1.055,
            "time_in_force": "gtc",            
         },
    }
    r = requests.post(APCA_API_ORDERS_URL, headers=headers, json=data)
    return json.loads(r.content)

# def get_asset(symbol):
#     r = requests.get(APCA_API_BASE_URL + f'/v2/assets/{symbol}', headers=headers)
#     return json.loads(r.content)

# def create_watchlist():
#     data = {
#         'name': 'ml',
#         #'symbols': ['MRNA', 'SPY', "QQQ", "IWM", "XLF", "XLK", "XOP", "XLP"]
#         "symbols": ['82d8c472-9b28-4b80-96c9-b3d97999a393'],
#     }
#     r = requests.post(APCA_API_ORDERS_URL, headers=headers, json=data)
#     return json.loads(r.content)

# def get_quote_snapshot(symbol):
#     requests.get("https://data.alpaca.markets/v2/stocks/{symbol}/snapshot", headers=headers)
#     r = requests.post(APCA_API_ORDERS_URL, headers=headers, json=data)
#     return json.loads(r.content)

def get_all_assets():
    r = requests.get(APCA_API_BASE_URL + '/v2/assets', headers=headers)
    return json.loads(r.content)

def get_quote_snapshots(symbols):
    r = requests.get("https://data.alpaca.markets/v2/stocks/snapshots", headers=headers, params={"symbols":",".join(symbols)})    
    return json.loads(r.content)

def init(symbols):
    all_assets = get_all_asset(symbol)
    #TODO: What happens when signal conflicts with existing position
    quotes_snapshots = get_quote_snapshots(symbols)

# symbol, action    
def process_signal(symbol, action):
    asset = [a for a in all_assets if a['symbol'] == symbol ]
    if not len(assets) == 1: return
    asset = asset[0]
    is_shortable = asset['easy_to_borrow']
    is_shortable = asset['shortable']
    
    close = quotes_snapshots[symbol]["dailyBar"]["c"]
    qty = int(amount_to_invest / close)


#%%
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--key-id', help='APCA_API_KEY_ID')
    parser.add_argument('--secret-key', help='APCA_API_SECRET_KEY')
    parser.add_argument('--base-url')
    args = parser.parse_args()

    #run({k: v for k, v in vars(args).items() if v is not None})

if __name__ == '__main__':
    main()
#req = requests.Request('GET', APCA_API_BASE_URL)
#r = req.prepare()

#%%
# account['buying_power']