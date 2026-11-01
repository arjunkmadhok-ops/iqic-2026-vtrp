import pandas as pd, numpy as np, itertools, sys
from multiprocessing import Pool
from engine import *
px = load('../data/etf.csv'); rets = px.pct_change(fill_method=None)
IS=('2002-08-01','2015-12-31')
RP=dict(top_n=11, trend=False, cap=0.2)
variants = {
 'rp_vt10': RP,
 'rp_vt10_band2': {**RP, 'band':0.02},
 'rp_vt10_dd5': {**RP, 'dd_breaker':0.05},
 'rp_vt10_dd8': {**RP, 'dd_breaker':0.08},
 'rp_vt10_cap15': {**RP, 'cap':0.15},
 'rp_vt10_cap30': {**RP, 'cap':0.30},
}
def one(k):
    r,W,T = run(px, variants[k], start='2000-01-03', end=IS[1])
    r = r[r.index>=IS[0]]; b=rets[BENCH].reindex(r.index).fillna(0); rf=rets[CASH].reindex(r.index).fillna(0)
    s = stats(r,b,rf); s['turn']=T[T.index>=IS[0]].sum()/(len(r)/252); s['eq']=1-W[CASH].loc[r.index].mean()
    return k,s
if __name__=='__main__':
    with Pool(8) as p: res=dict(p.map(one, variants))
    r=rets.loc['2002-08-01':IS[1]]
    res['SPY']=stats(r[BENCH],r[BENCH],r[CASH])
    print(pd.DataFrame(res).T.round(3).to_string())
