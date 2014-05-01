__author__ = 'jchristensen'

import pandas.io.data as web
from datetime import timedelta, datetime, date
import numpy as np

end = date.today()
#end = datetime(2014,3,1,0,0,0)
start = end - timedelta(days=800)

stocks = sorted(['PRF','VEU','BLV','VNQ','EMB','DBC','DBA','VWO','GLD',
'CRQ.TO', 'EMB', 'VTIP', 'VNQI', 'EWJ', 'LQD'])
#stocks = sorted(['RWX', 'PCY', 'WIP', 'EFA', 'HYG', 'EEM', 'LQD', 'VNQ', 'TIP', 'VTI', 'DBC', 'GLD', 'DBA', 'TLT'])
#stocks = sorted(['FTS.TO','CU.TO','ACO-X.TO','TRI.TO','ENB.TO','IMO.TO','CNR.TO',
#                'ESI.TO','EMP-A.TO','TRP.TO','SAP.TO','SNC.TO','REF-UN.TO','CNQ.TO',
#                'MHR.TO','SU.TO','RBA.TO','PSI.TO','HCG.TO','MRU.TO','CMG.TO','SCL.TO',
#                'RCI-B.TO','SJR-B.TO','CCL-B.TO','IFC.TO','EMA.TO','CCA.TO','LB.TO','GS.TO',
#                'GLN.TO','LAS-A.TO','THI.TO','HLF.TO','BYD-UN.TO','PJC-A.TO','BCE.TO','ESL.TO','FNV.TO'])

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
