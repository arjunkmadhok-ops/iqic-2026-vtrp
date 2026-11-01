import pandas as pd, numpy as np, sys, json
sys.path.insert(0,'../alt/code')
import engine
from engine import load, run
import importlib.util
spec=importlib.util.spec_from_file_location('alt','../alt/code/IQIC_2026_Sector_Alpha_Backtest.py'); alt=importlib.util.module_from_spec(spec); spec.loader.exec_module(alt)
px=load('../data/etf.csv'); rets=px.pct_change(fill_method=None)
SEC=["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
engine.SECTORS=SEC+['IEF','GLD']
B=dict(top_n=99, trend=False, cap=0.25, target_vol=0.10, band=0.02, dd_breaker=0.08, cost=0.001)
r,W,T=run(px,B,start='2005-01-03')
def navframe(rr, a, b):
    x=rr.loc[a:b]; nav=(1+x).cumprod()
    return pd.DataFrame({'nav':nav,'SPY_ret':rets.SPY.reindex(nav.index)})
alt_def=pd.read_csv('../alt/res_default/nav.csv',index_col=0,parse_dates=True)
alt_long=pd.read_csv('../alt/res_long/nav.csv',index_col=0,parse_dates=True)
rows={}
for name,(a,b) in {'2019-22':('2019-01-01','2022-12-31'),'2023-Oct26':('2023-01-01','2026-12-31'),'2005-15':('2005-08-01','2015-12-31'),'2016-Oct26':('2016-01-01','2026-12-31')}.items():
    ours=navframe(r,a,b)
    src=alt_def if name in('2019-22','2023-Oct26') else alt_long
    theirs=src.loc[a:b]
    spy=navframe(rets.SPY.fillna(0),a,b)
    for lab,nf in [('declared',ours),('sector_alpha',theirs),('SPY',spy)]:
        m=alt.metrics(nf,rf=0.03)
        w=alt.fifteen_day_tests(nf)
        rows[(name,lab)]={'Sharpe':m['sharpe_daily_annualized'],'alpha':m['jensen_alpha_daily_annualized'],'beta':m['beta'],'Treynor':m['treynor_daily_annualized'],'CAGR':m['annual_return_CAGR'],'MaxDD':m['max_drawdown'],
            'win15_vs_SPY':(w.excess_15d>0).mean() if lab!='SPY' else np.nan,'loss_when_SPY_down':w.loc[w.SPY_15d<0,'strategy_15d'].mean()}
print(pd.DataFrame(rows).T.round(3).to_string())
