import numpy as np, pandas as pd
from scipy import stats
from combine import wide
from hyb import rollwide, combo, HY
o=pd.read_csv('out_ts_prev20.csv'); T=wide(o).loc['test']; V=rollwide('roll_ts_prev.csv')
P=pd.DataFrame(index=T.index); P['obs']=T.obs
for m in ['XGB','LSTM','TCN']: P[m]=T[m]
for n,ms in HY.items(): P[n]=combo(T,ms,V)[0]
B=pd.read_csv('baseline_preds_theilsen.csv').set_index(['District','Year']); Bo=pd.read_csv('baseline_preds_ols.csv').set_index(['District','Year'])
for c in ['Trend only (Theil–Sen)','Trend only (recent window 2011–2020)','Ridge','Elastic net','Random forest','SVR (RBF)','AR(1) residual','Persistence']: P[c]=B[c]
P['Trend only (OLS 2001–2020)']=Bo['Trend only (OLS 2001–2020)']; P['Climatology']=B['Climatology']
P.to_csv('final_preds_re.csv')
def dm(d):
    n=len(d); s=d.mean()/np.sqrt(d.var(ddof=0)/n)*np.sqrt((n-1)/n); return s,2*stats.t.sf(abs(s),n-1)
def bh(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); q=np.empty(m); ps=p[o]*m/np.arange(1,m+1); ps=np.minimum.accumulate(ps[::-1])[::-1]; q[o]=np.minimum(ps,1); return q
rows=[]
yr=P.index.get_level_values('Year')
for c in P.columns[2:]:
    d=(P.XGB-P.obs)**2-(P[c]-P.obs)**2
    s,p=dm(d.values); dy=d.groupby(yr).mean().values; s5,p5=dm(dy)
    rows.append(dict(model=c,DM=s,p=p,DMy=s5,py=p5,yrs_better=int((dy<0).sum())))
R=pd.DataFrame(rows); R['q']=bh(R.p); R['qy']=bh(R.py)
pd.set_option('display.width',200); print(R.round(3).to_string())
R.to_csv('dm_year.csv',index=False)
