import pandas as pd, numpy as np
from multiprocessing import Pool
import engine
from engine import load, run, stats
px = load('../data/etf.csv'); rets = px.pct_change(fill_method=None)
SEC = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
BASE = dict(top_n=99, trend=False, cap=0.2, target_vol=0.10, band=0.02, dd_breaker=0.08)
V = {
 'A_sectors': (SEC, {}),
 'B_sec_ief_gld': (SEC+['IEF','GLD'], {'cap':0.25}),
 'C_sec_gld': (SEC+['GLD'], {}),
 'D_sec_ief': (SEC+['IEF'], {'cap':0.25}),
 'E_sec_tlt_gld': (SEC+['TLT','GLD'], {'cap':0.25}),
}
def job(k):
    u,p = V[k]; engine.SECTORS = u
    r,W,T = run(px, {**BASE, **p}, start='2005-01-03')
    W.to_csv(f'w_{k}.csv')
    return k, r, T
if __name__=='__main__':
    with Pool(2) as pool: out = pool.map(job, list(V))
    d = pd.DataFrame({k:r for k,r,_ in out}); d['spy']=rets.SPY; d['rf']=rets.SHY; d['qqq']=rets.QQQ
    d = d.loc['2005-08-01':].fillna(0); d.to_csv('multi_daily.csv')
    res={}
    for per,(a,b) in {'IS 05-15':('2005-08-01','2015-12-31'),'OOS 16-26':('2016-01-01','2026-12-31')}.items():
        x=d.loc[a:b]
        for k in list(V)+['spy']:
            s=stats(x[k],x.spy,x.rf); res[(per,k)]=s
    print(pd.DataFrame(res).T.round(3).to_string())
    idx=d.index; rows=[]
    for s in range(0,len(idx)-15,5):
        x=d.iloc[s:s+15]; rec={'start':idx[s]}
        for k in list(V)+['spy','qqq']:
            st=stats(x[k],x.spy,x.rf); rec[k+'|sh']=st['Sharpe']; rec[k+'|al']=st['Alpha']; rec[k+'|tr']=st['Treynor']; rec[k+'|ret']=(1+x[k]).prod()-1; rec[k+'|beta']=st['Beta']
        rows.append(rec)
    R=pd.DataFrame(rows).set_index('start'); R.to_csv('multi_windows.csv')
    for per,(a,b) in {'IS':('2005','2015'),'OOS':('2016','2026')}.items():
        x=R.loc[a:b]; print(per, len(x),'windows')
        for k in V:
            print(f"  {k:15s} Sh>SPY {(x[k+'|sh']>x['spy|sh']).mean():.2f}  Sh>QQQ {(x[k+'|sh']>x['qqq|sh']).mean():.2f}  alpha>0 {(x[k+'|al']>0).mean():.2f}  medSh {x[k+'|sh'].median():.2f} (spy {x['spy|sh'].median():.2f})  medBeta {x[k+'|beta'].median():.2f}  ret<0 {(x[k+'|ret']<0).mean():.2f} (spy {(x['spy|ret']<0).mean():.2f})")
