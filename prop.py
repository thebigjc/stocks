
__author__ = 'jchristensen'

import pandas.io.data as web
import pandas as pd
from datetime import timedelta, datetime, date
import numpy as np
import scipy as sp
import sys
import os, os.path
from scipy.optimize import brute
from scipy.stats import t
from math import sqrt

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


import warnings
warnings.filterwarnings('error')

def max_dd(x):
    cum_prod = x.cumprod()
    i = np.argmax(np.maximum.accumulate(cum_prod) - cum_prod)
    j = np.argmax(cum_prod[:i])
    
    return float(x.loc[j:i].prod()-1)*100.0

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

def run_prop(STOCK, BOND, LOOKBACK, dump=False):
    end = date.today()
    #end = datetime(2014,5,1,0,0,0)
    start = end - timedelta(days=3000)

    stocks = [STOCK, BOND]

    print (start,end)

    dr = load_stocks(stocks, start, end)

    adj_close = dr['Adj Close'].fillna(method='pad')

    stock = adj_close[STOCK].pct_change() + 1
    bond = adj_close[BOND].pct_change() + 1

    delta = stock - bond
    delta_mean = pd.rolling_mean(delta, LOOKBACK)
    delta_std = pd.rolling_std(delta, LOOKBACK)

    d = pd.DataFrame(index=delta.index)
    d['delta'] = delta
    d['delta_mean'] = delta_mean
    d['delta_std'] = delta_std
    d['ir'] = sqrt(LOOKBACK) * d['delta_mean'] / d['delta_std']

    t_df = t(LOOKBACK)

    d['p'] = t_df.cdf(d.ir)

    d['t'] = np.nan
    d['t'][d['p'] < 0.4] = BOND
    d['t'][d['p'] > 0.6] = STOCK
    d = d.fillna(method='ffill').dropna()
    d['stock'] = stock
    d['bond'] = bond
    d['strat'] = stock
    d.loc[d['t'] == BOND,'strat'] = bond

    print stocks
    start = d.index[0].to_datetime()
    end = d.index[-1].to_datetime()

    print start
    years = (end-start).days/365.0

    print "Stock CAGR:", d['stock'].prod() ** (1/years)
    print "Strat CAGR:", d['strat'].prod() ** (1/years)
    print "Better dates:", (d['strat'] > d['stock']).sum()
    better = d.loc[d['strat'] > d['stock'], 'strat'].prod()
    print "Better prod:", better
    print "Worse dates:", (d['strat'] < d['stock']).sum()
    worse = d.loc[d['strat'] < d['stock'], 'strat'].prod()

    print "Worse prod:", worse
    print "Lift:", better * worse
    trades = (d['t'] <> d.shift()['t']).sum()
    print "Trades:", trades
    print "Trades / Year:", trades / years
    print "Current:", d['t'].iloc[-1]

    print "Strat MaxDD %0.2f%%" % max_dd(d['strat'])
    print "Stock MaxDD %0.2f%%" % max_dd(d['stock'])

    x = d['t'] <> d.shift()['t']
    last = x[x].index[-1]
    print "Last Trade:", last
    print "Strat return since last: %0.2f%%" % (float(d['strat'].loc[last:].prod()-1) * 100.0)
    print "Stock return since last: %0.2f%%" % (float(d['stock'].loc[last:].prod()-1) * 100.0)

if __name__ == '__main__':
    run_prop('CDZ.TO', 'XLB.TO', 60)
    run_prop('RSP', 'TLT', 60)

