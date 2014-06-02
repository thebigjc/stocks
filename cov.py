__author__ = 'jchristensen'

import pandas.io.data as web
import pandas as pd
from datetime import timedelta, datetime, date
import numpy as np
import scipy as sp
import sys
import os, os.path

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


def calc_sharpe(weights, returns):
    #reg = np.power(weights, 2).sum()
    reg = weights.sum()
    if reg == 0.0:
        return 0.0

    l = len(weights[weights > 0])

    weights = weights / weights.sum()
    port_rets = (returns * weights).sum(1)
    mean = port_rets.mean()
    if np.isnan(mean):
        print reg
    std = port_rets.std()
    sharpe = mean / std
#    print sharpe, mean, std, reg, l

    l = l - 6
    l *= l

    return 0.1 * reg + 0.1 * l - sharpe

def get_opt_holdings_opt(returns):
    import scipy.optimize as opt
    K = len(returns.columns)
    init_weights = np.ones(K) / K
    bounds = zip(np.zeros(K), np.ones(K))
    result = opt.fmin_l_bfgs_b(calc_sharpe, init_weights,
                bounds = bounds,
                args=(returns,), approx_grad=True, factr=1e7, pgtol=1e-7,
                iprint=0)

    print result

    return pd.Series(result[0], index=returns.columns)

if len(sys.argv) > 1:
    syms = open(sys.argv[1], 'rU')
    syms = [x.strip() for x in syms.readlines() if len(x.strip())]
else:
    syms = open('div.csv', 'rU')
    syms = [x.strip().replace('.', '-')+".TO" for x in syms.readlines() if len(x.strip())]

end = date.today()
#end = datetime(2014,2,10,0,0,0)
start = end - timedelta(days=300)

#stocks = sorted(['BMO.TO', 'POT.TO', 'CBY.TO', 'G.TO', 'BCE.TO', 'TCL-A.TO', 'VSN.TO'])
stocks = sorted(syms)

dr = load_stocks(stocks, start, end) 

adj_close = dr['Adj Close'].fillna(method='pad')
sma200 = adj_close.iloc[-200:].mean() < adj_close.iloc[-1]
adj_close = adj_close[sma200[sma200].index.values]

momentum = adj_close.iloc[-1] / adj_close.iloc[-22*3] - 1
stocks = momentum[momentum > 0].index.values

adj_close = adj_close[stocks]

returns = adj_close.pct_change()+1

log_returns = np.log(returns - 0.01/252)

weights = get_opt_holdings_opt(log_returns[-22*3:])
weights = weights / weights.sum()
weights = weights[weights > 0.01].order()

print weights

COUNT = 12

stocks_w = weights[-COUNT:].index.values

price = dr['Close'].iloc[-1]

price = price[stocks_w]

print price

stocks = momentum[stocks_w].order()[-COUNT/2:].index.values
price = price[stocks]

PORT_SIZE = 103415.00 - 3540.00

port = PORT_SIZE * 1. / len(price) / price

shares_low = sp.floor(port / 50)*50
shares_high = sp.ceil(port / 50)*50

port = pd.DataFrame(index=shares_low.index)
port['shares_low'] = shares_low
port['shares_high'] = shares_high
port['price'] = price
port['book_low'] = port.shares_low * port.price
port['book_high'] = port.shares_high * port.price
port['momentum'] = momentum

def bitfield(n):
    return [1 if digit=='1' else 0 for digit in bin(n)[2:]]

min_diff = PORT_SIZE
best_bf = [0] * len(port)

for x in xrange(2 ** len(port)):
    bf = bitfield(x)
    bf = bf + [0,] * (len(port) - len(bf))
    port['field'] = bf
    diff = PORT_SIZE - (port.book_low * (1-port.field) + port.book_high * port.field).sum()
    if abs(diff) < min_diff:
        min_diff = diff
        best_bf = bf
        print x, diff, bf

port['field'] = best_bf
port['shares'] = port.shares_low * (1-port.field) + port.shares_high * port.field
port['book'] = port.shares * port.price

print port.sort()
print port.book.sum()
