import pandas as pd, numpy as np
from multiprocessing import Pool
import engine
from engine import load, run, stats, target_weights
px = load('../data/etf.csv'); rets = px.pct_change(fill_method=None)
SEC = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
U = SEC+['IEF','GLD']
B = dict(top_n=99, trend=False, cap=0.25, target_vol=0.10, band=0.02, dd_breaker=0.08)
def job(a):
    name,P=a; engine.SECTORS=U
    r,W,T=run(px,{**B,**P},start='2005-01-03'); return name,r,W,T
if __name__=='__main__':
    jobs=[('final',{}),('cost20',{'cost':0.002}),('no_dd',{'dd_breaker':None}),('no_vt',{'target_vol':None})]
    for tv in [0.08,0.10,0.12,0.14]:
        for vw in [21,63,126]:
            if (tv,vw)!=(0.10,63): jobs.append((f'tv{int(tv*100)}|vw{vw}',{'target_vol':tv,'vol_win':vw}))
    with Pool(2) as p: out=p.map(job,jobs)
    res={n:(r,W,T) for n,r,W,T in out}
    d=pd.DataFrame({n:r for n,(r,_,_) in res.items()}); d['spy']=rets.SPY; d['rf']=rets.SHY; d=d.loc['2005-08-01':].fillna(0)
    d.to_csv('B_daily.csv'); res['final'][1].to_csv('B_weights.csv')
    rows={}
    for n in res:
        for per,(a,b) in {'IS':('2005-08-01','2015-12-31'),'OOS':('2016-01-01','2026-12-31'),'Full':('2005-08-01','2026-12-31')}.items():
            x=d.loc[a:b]; s=stats(x[n],x.spy,x.rf); rows[(n,per)]=s['Sharpe']
    print(pd.Series(rows).unstack().round(3).to_string())
    x=d; print('Full final', {k:round(v,3) for k,v in stats(x.final,x.spy,x.rf).items()}); print('Full spy', {k:round(v,3) for k,v in stats(x.spy,x.spy,x.rf).items()})
    W=res['final'][1].loc['2005-08-01':]; T=res['final'][2].loc['2005-08-01':]
    print('avg weights', W.mean().round(3).to_dict())
    print('turnover/yr', T.sum()/(len(W)/252))
    # stress windows
    for nm,(a,b) in {'GFC 2007-10..2009-03':('2007-10-09','2009-03-09'),'COVID Feb19-Mar23 2020':('2020-02-19','2020-03-23'),'2022 bear Jan3-Oct12':('2022-01-03','2022-10-12'),'Apr 2025 tariff Feb19-Apr8':('2025-02-19','2025-04-08')}.items():
        x=d.loc[a:b]; print(nm, 'vtsm', round((1+x.final).prod()-1,3), 'spy', round((1+x.spy).prod()-1,3))
    # current weights as of last date
    engine.SECTORS=U
    w=target_weights(px,rets,px.index[-1],{**engine.DEFAULT,**B})
    print('as of',px.index[-1].date()); print(w.round(4).to_string())
    win=rets.iloc[-63:]; print('63d vol', (win[U].std()*np.sqrt(252)).round(3).to_dict())
