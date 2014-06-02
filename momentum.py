__author__ = 'jchristensen'

import pandas.io.data as web
import pandas as pd
from datetime import timedelta, datetime, date
import numpy as np
import scipy as sp
import sys
import os, os.path
from scipy.optimize import brute

end = date.today()
#end = datetime(2014,5,1,0,0,0)
start = end - timedelta(days=800)

def load_stocks(stocks, start, end):
    fn = "cache/%s-%s-%s.pkl" % ('_'.join(stocks), str(start), str(end))
    if os.path.isfile(fn) and os.access(fn, os.R_OK):
        print "Cached"
        return pd.read_pickle(fn)

    print "Fetching"
    pnl = web.DataReader(stocks, "yahoo", start, end)
    print "Saving"
    pnl.to_pickle(fn)

    return pnl

def slippage(vals, price, closes):
    total = vals * 50 * closes
    if total.sum() > price.sum():
        return float("inf")

    obj = (total - price)**2
    #print vals, obj.sum()

    return obj.sum()

if len(sys.argv) < 2:
    stocks = ['BLV', # Vanguard Long-Term Bond ETF
            'XIC.TO', # iShares S&P/TSX Capped Composite Index ETF
        #    'ZLB.TO', # BMO Low Volatility Canadian Equity ETF
        #'XMV.TO', # iShares MSCI Canada Minimum Volatility Index ETF
        'DBA', # DB Agriculture Fund
        'GSP', # 
        'EMB', # iShares J.P. Morgan USD Emerging Markets Bond ETF
        'EWJ', # iShares MSCI Japan ETF
        'GLD', # SPDR Gold Trust
        'LQD', # iShares iBoxx $ Investment Grade Corporate Bond ETF
        'PRF', # FTSE RAFI US 1000 Portfolio
        'RWX', # SPDR DJ Wilshire Intl Real Estate
        'VCIT', # Intermediate-Term Corporate Bond Index Fund
        'VEU', # FTSE All World Ex US ETF
        'VNQ', # REIT ETF
        'TIP', # TIPS Bond ETF
        'VWO' # Emerging Markets ETF
        ]
else:
    stocks = [x.strip() for x in open(sys.argv[1]).readlines()]
    print stocks

print (start,end)

dlr = load_stocks(['DLR.TO'], start, end)

dr = load_stocks(stocks, start, end)

dlr = dlr['Adj Close'].fillna(method='pad') / 10
adj_close = dr['Adj Close'].fillna(method='pad')
close = dr['Close'].fillna(method='pad')

not_ca = [s for s in stocks if not s.endswith('.TO')]

#usd = adj_close[not_ca] * dlr

sma200 = adj_close[-200:].mean()

sma200 = sma200 < adj_close.iloc[-1]

for s in not_ca:
    adj_close[s] *= dlr
    close[s] *= dlr

returns = np.log(adj_close.shift(1) / adj_close)
cor = returns.corr()

x = None

for i in (1, 3, 6, 12):
    a = np.log((adj_close.iloc[-1] / adj_close.iloc[-i*22]))
    if x is None:
        x = a
    else:
        x += a

x /= 4
x = np.exp(x)

x = x[sma200]

x = x.order()

x = x[-6:]

norm = x/x.sum()

selected_stocks = x.index.values

PORT_SIZE = 283251.57

close = close.iloc[-1][selected_stocks]

price = PORT_SIZE * norm
shares = price / close

shares_high = sp.ceil(shares / 50)
shares_low = sp.floor(shares / 50)

ranges = [slice(int(l-1), int(h+1)) for l,h in zip(shares_low, shares_high)]
print ranges

shares = brute(slippage, ranges, (price, close), finish=None)

port = pd.DataFrame(index=price.index)
port['lots'] = shares
port['close'] = close
port['price'] = price
port['shares'] = port.lots * 50
port['book'] = port.shares * port.close
port['slippage'] = port.book - port.price

print port
print port.book.sum(), PORT_SIZE
