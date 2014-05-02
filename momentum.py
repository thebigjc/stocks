__author__ = 'jchristensen'

import pandas.io.data as web
from datetime import timedelta, datetime, date
import numpy as np

end = date.today()
#end = datetime(2014,3,1,0,0,0)
start = end - timedelta(days=800)

stocks = ['BLV', # Vanguard Long-Term Bond ETF
    'XIC.TO', # iShares S&P/TSX Capped Composite Index ETF
    #    'ZLB.TO', # BMO Low Volatility Canadian Equity ETF
    #'XMV.TO', # iShares MSCI Canada Minimum Volatility Index ETF
    'DBA', # DB Agriculture Fund
    'DBC', # DB Commodity Index Tracking Fund
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

print (start,end)

dlr = web.DataReader('DLR.TO', "yahoo", start, end)

dr = web.DataReader(stocks, "yahoo", start, end)

dlr = dlr['Adj Close'].fillna(method='pad') / 10
adj_close = dr['Adj Close'].fillna(method='pad')

not_ca = [s for s in stocks if not s.endswith('.TO')]

#usd = adj_close[not_ca] * dlr

sma200 = adj_close[-200:].mean()

sma200 = sma200 < adj_close.iloc[-1]

for s in not_ca:
    adj_close[s] *= dlr

returns = np.log(adj_close.shift(1) / adj_close)
cor = returns.corr()

x = None

for i in (1, 3, 6, 12):
    a = np.log((adj_close.iloc[-1] / adj_close.iloc[-12*22]))
    if x is None:
        x = a
    else:
        x += a

x /= 4
x = np.exp(x)

x = x[sma200]

x = x.order()

x = x[-6:]

print cor[np.abs(cor) > 0.9]

#cor = cor[x.index]

#print(cor[cor.abs() > 0.90])

print(x)
