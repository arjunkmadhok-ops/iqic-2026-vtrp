import pandas as pd, numpy as np
from multiprocessing import Pool
import engine
from engine import load, run, stats
px = load('../data/etf.csv'); rets = px.pct_change(fill_method=None)
SEC = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
engine.SECTORS = SEC
def tw(px, rets, t, P):
    loc = px.index.get_loc(t)
    if loc < 260: return None
    win = rets.iloc[loc-19:loc+1]
    spy = px['SPY']
    m20s = spy.iloc[loc]/spy.iloc[loc-20]-1; m5s = spy.iloc[loc]/spy.iloc[loc-5]-1
    sc = {}
    for s in SEC:
        p = px[s]
        if np.isnan(p.iloc[loc-20]) or win[s].isna().any() or np.isnan(p.iloc[loc-130]): continue
        m20 = p.iloc[loc]/p.iloc[loc-20]-1-m20s; m5 = p.iloc[loc]/p.iloc[loc-5]-1-m5s
        sc[s] = (0.6*m20+0.4*m5)/win[s].std()
    pick = sorted(sc, key=sc.get, reverse=True)[:P['n']]
    w = pd.Series(0.0, index=SEC+['SHY'])
    if P.get('trend') and spy.iloc[loc] < spy.iloc[loc-199:loc+1].mean():
        w['SHY'] = 1.0; return w
    vol = win[pick].std()*np.sqrt(252)
    ws = (1/vol)/(1/vol).sum()
    for _ in range(20):
        over = ws > P['cap']+1e-12
        if not over.any(): break
        ex = (ws[over]-P['cap']).sum(); ws[over] = P['cap']; ws[~over] += ex*ws[~over]/ws[~over].sum()
    scale = 1.0
    if P.get('tv'):
        cov = rets.iloc[loc-62:loc+1][pick].cov()*252
        scale = min(1.0, P['tv']/np.sqrt(ws.values@cov.values@ws.values))
    w[pick] = ws*scale; w['SHY'] = 1-w[pick].sum()
    return w
engine.target_weights = tw
V = {'top4': dict(n=4, cap=0.30), 'top4_trend': dict(n=4, cap=0.30, trend=True),
     'top4_trend_vt10': dict(n=4, cap=0.30, trend=True, tv=0.10), 'top3_trend': dict(n=3, cap=0.40, trend=True)}
def job(k):
    r,W,T = run(px, {**V[k], 'band':0.02, 'cost':0.0005}, start='2003-01-02')
    return k, r, T
if __name__=='__main__':
    with Pool(2) as p: out = p.map(job, list(V))
    d = pd.DataFrame({k:r for k,r,_ in out}); turn = {k:T for k,_,T in out}
    B = pd.read_csv('B_daily.csv', index_col=0, parse_dates=True)['final']
    d['declared'] = B; d['spy'] = rets.SPY; d['rf'] = rets.SHY; d = d.loc['2005-08-01':].fillna(0)
    rows = {}
    for per,(a,b) in {'2005-15':('2005-08-01','2015-12-31'),'2016-26':('2016-01-01','2026-12-31')}.items():
        x = d.loc[a:b]
        for k in list(V)+['declared','spy']:
            s = stats(x[k], x.spy, x.rf)
            if k in turn: s['turn'] = turn[k].loc[a:b].sum()/(len(x)/252)
            rows[(per,k)] = s
    print(pd.DataFrame(rows).T.round(3).to_string())
    idx = d.index; wins = []
    for s0 in range(0, len(idx)-15, 5):
        x = d.iloc[s0:s0+15]; rec = {'start': idx[s0]}
        for k in list(V)+['declared','spy']:
            rec[k] = stats(x[k], x.spy, x.rf)['Sharpe']
        wins.append(rec)
    R = pd.DataFrame(wins).set_index('start')
    for per,(a,b) in {'2005-15':('2005','2015'),'2016-26':('2016','2026')}.items():
        x = R.loc[a:b]
        print(per, {k: round((x[k] > x.spy).mean(), 2) for k in list(V)+['declared']})
