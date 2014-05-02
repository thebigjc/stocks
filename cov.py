__author__ = 'jchristensen'

import pandas.io.data as web
import pandas as pd
from datetime import timedelta, datetime, date
import numpy as np
import scipy as sp
import sys

def calc_sharpe(weights, returns):
    #reg = np.power(weights, 2).sum()
    reg = weights.sum()
    if reg == 0.0:
        return 0.0

    weights = weights / weights.sum()
    port_rets = (returns * weights).sum(1)
    mean = port_rets.mean()
    if np.isnan(mean):
        print reg
    std = port_rets.std()
    sharpe = np.sqrt(len(weights)) * (mean / std)
    print sharpe, mean, std, reg

    return -sharpe + 0.1 * (reg-1) * (reg-1)

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

dr = web.DataReader(stocks, "yahoo", start, end)

adj_close = dr['Adj Close'].fillna(method='pad')
sma200 = adj_close[-200:].mean() < adj_close.iloc[-1]

returns = adj_close.pct_change()+1
print returns.columns[sma200]

log_returns = np.log(returns - 0.01/252)

weights = get_opt_holdings_opt(log_returns[-22*3:])
weights = weights / weights.sum()
#weights = weights[weights > 0.01].order()
weights = weights.order()

print weights

COUNT = 12

stocks_w = weights[-COUNT:].index.values

price = dr['Close'].iloc[-1]

price = price[stocks_w]
momentum = dr['Adj Close'][stocks_w].iloc[-1] / dr['Adj Close'][stocks_w].iloc[-22*3] - 1
#momentum = momentum[momentum > 0]

stocks = momentum.order()[-COUNT/2:].index.values
price = price[stocks]

PORT_SIZE = 100000

port = PORT_SIZE * 1. / len(price) / price

shares = sp.floor(port / 50 + 0.5)*50

port = pd.DataFrame(index=shares.index)
port['shares'] = shares
port['price'] = price
port['book'] = port.shares * port.price
port['momentum'] = momentum

print port

print log_returns[stocks_w].corr()
