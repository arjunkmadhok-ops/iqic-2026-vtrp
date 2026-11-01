import pandas as pd, numpy as np
from engine import stats, SECTORS
d=pd.read_csv('final_daily.csv',index_col=0,parse_dates=True)
W=pd.read_csv('final_weights.csv',index_col=0,parse_dates=True)
T=pd.read_csv('final_turn.csv',index_col=0,parse_dates=True).iloc[:,0]
out={}
for per,(a,b) in {'IS 2002-15':('2002-08-01','2015-12-31'),'OOS 2016-26':('2016-01-01','2026-12-31'),'Full':('2002-08-01','2026-12-31')}.items():
    x=d.loc[a:b]
    for k in ['vtsm','spy','ew','qqq']:
        if k=='qqq' and per!='OOS 2016-26': pass
        out[(per,k)]=stats(x[k],x['spy'],x['rf'])
t=pd.DataFrame(out).T; print(t.round(3).to_string())
print('last date', d.index[-1])
oos=W.loc['2016':]
print('avg equity OOS', 1-oos['SHY'].mean(), 'min',1-oos['SHY'].min(),'max',1-oos['SHY'].max())
print('turnover/yr OOS', T.loc['2016':].sum()/ (len(d.loc['2016':])/252))
# rolling 15-day non-overlapping windows... use all starting Mondays (overlapping, step 5)
rows=[]
idx=d.index
for s in range(0,len(idx)-15,5):
    x=d.iloc[s:s+15]
    rec={'start':idx[s]}
    for k in ['vtsm','spy','ew','qqq']:
        st=stats(x[k],x['spy'],x['rf'])
        rec[k+'_sh']=st['Sharpe']; rec[k+'_ret']=(1+x[k]).prod()-1; rec[k+'_alpha']=st['Alpha']; rec[k+'_beta']=st['Beta']
        rec[k+'_vol']=st['Vol']
    rows.append(rec)
R=pd.DataFrame(rows).set_index('start')
R.to_csv('windows.csv')
for per,(a,b) in {'Full':('2002','2026'),'OOS':('2016','2026')}.items():
    x=R.loc[a:b]; n=len(x)
    print(per,'windows',n)
    for k in ['spy','ew','qqq']:
        if k=='qqq': x2=x.dropna(subset=['qqq_sh'])
        else: x2=x
        print(f"  vtsm Sharpe > {k}: {(x2.vtsm_sh>x2[k+'_sh']).mean():.3f}")
    print(f"  vtsm 3wk alpha>0: {(x.vtsm_alpha>0).mean():.3f}   spy ret<0 windows: {(x.spy_ret<0).mean():.3f}")
    dn=x[x.spy_ret<0]; up=x[x.spy_ret>0]
    print(f"  down windows: vtsm mean ret {dn.vtsm_ret.mean():.4f} spy {dn.spy_ret.mean():.4f}; up windows vtsm {up.vtsm_ret.mean():.4f} spy {up.spy_ret.mean():.4f}")
    print(f"  median 3wk vol vtsm {x.vtsm_vol.median():.3f} spy {x.spy_vol.median():.3f}; median beta {x.vtsm_beta.median():.2f}")
    print('  3wk return percentiles vtsm', x.vtsm_ret.quantile([.05,.5,.95]).round(4).tolist(), 'spy', x.spy_ret.quantile([.05,.5,.95]).round(4).tolist())
    # Nov windows
nov=R[(R.index.month==11)&(R.index.day<=7)]
print(nov[['vtsm_ret','spy_ret','vtsm_sh','spy_sh']].round(3).tail(12))
