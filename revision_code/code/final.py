import json, numpy as np, pandas as pd
from scipy import stats
from pipe import metrics, rmse, DIST, base_df, fit_trend
from combine import wide, PAIRS
MAIN='ts_prev20'; ROLLV='roll_ts_prev'; ROLLT='roll_ts_prev_test'
def rollwide(f):
    R=pd.read_csv(f).groupby(['District','Year','model']).agg(pred=('pred','mean'),obs=('obs','first')).reset_index()
    V=R.pivot_table(index=['District','Year'],columns='model',values='pred'); V['obs']=R.groupby(['District','Year']).obs.first(); return V
RV=rollwide(f'roll_{ROLLV}.csv') if not ROLLV.startswith('roll_') else rollwide(f'{ROLLV}.csv')
out=pd.read_csv(f'out_{MAIN}.csv'); W=wide(out); T=W.loc['test'].copy()
# district weights from rolling pre-test 2014-2020
WT={}
def combo(F, ms, mode='district', V=RV):
    p=pd.Series(0.0,index=F.index)
    for d in DIST:
        g=V.loc[d]
        if mode=='district': inv=np.array([1/rmse(g[m]-g.obs) for m in ms])
        elif mode=='global': inv=np.array([1/rmse(V[m]-V.obs) for m in ms])
        elif mode=='val2':  # original 2-year validation weights
            vv=W.loc['val'].loc[d]; inv=np.array([1/rmse(vv[m]-vv.obs) for m in ms])
        else: inv=np.ones(len(ms))
        w=inv/inv.sum(); WT[(mode,'+'.join(ms),d)]=dict(zip(ms,w.round(3)))
        p.loc[d]=(F.loc[d][ms].values*w).sum(1)
    return p
HY={'XGB+LSTM':['XGB','LSTM'],'TCN+XGB':['TCN','XGB'],'LSTM+TCN':['LSTM','TCN'],'XGB+LSTM+TCN':['XGB','LSTM','TCN']}
P=pd.DataFrame(index=T.index); P['obs']=T.obs
for m in ['XGB','LSTM','TCN']: P[m]=T[m]
for n,ms in HY.items(): P[n]=combo(T,ms)
PV=pd.DataFrame(index=RV.index); PV['obs']=RV.obs
for m in ['XGB','LSTM','TCN']: PV[m]=RV[m]
for n,ms in HY.items(): PV[n]=combo(RV,ms)
# baselines (Theil-Sen protocol)
B=pd.read_csv('baseline_preds_theilsen.csv').set_index(['District','Year'])
Bo=pd.read_csv('baseline_preds.csv').set_index(['District','Year'])
P['Trend only (Theil–Sen)']=B['Trend only (Theil–Sen)']; P['Trend only (OLS)']=Bo['Trend only (OLS 2001–2020)']
P['Trend only (recent window)']=B['Trend only (recent window 2011–2020)']
for c in ['Ridge','Elastic net','Random forest','SVR (RBF)','AR(1) residual']: P[c]=B[c]
P['Persistence']=B['Persistence']; P['Climatology']=B['Climatology']
P.to_csv('final_preds.csv'); PV.to_csv('final_rollval_preds.csv')
tr=P['Trend only (Theil–Sen)']
rows=[]
for c in P.columns[1:]:
    m=metrics(P.obs,P[c]); m['Skill']=1-np.mean((P[c]-P.obs)**2)/np.mean((tr-P.obs)**2)
    m['PreTestRMSE']=rmse(PV[c]-PV.obs) if c in PV else np.nan; m['model']=c; rows.append(m)
T3=pd.DataFrame(rows).set_index('model'); print(T3.round(3).to_string()); T3.to_csv('T3.csv')
# seeds
ps=[]
for s in sorted(out.seed.unique()):
    if s==42: continue
    o=out[(out.seed==s)|(out.model=='XGB')]; Ws=wide(o).loc['test']
    r={'seed':s}
    for m in ['LSTM','TCN']: r[m]=metrics(Ws.obs,Ws[m])['R2']; r[m+'_RMSE']=metrics(Ws.obs,Ws[m])['RMSE']
    for n,ms in HY.items(): pp=combo(Ws,ms); r[n]=metrics(Ws.obs,pp)['R2']; r[n+'_RMSE']=metrics(Ws.obs,pp)['RMSE']
    ps.append(r)
S=pd.DataFrame(ps); S.to_csv('seeds.csv',index=False); print(S.describe().loc[['mean','std','min','max']].round(3).T.to_string())
# weighting modes
rows=[]
for n,ms in HY.items():
    for mode in ['district','global','equal','val2']:
        p=combo(T,ms,mode); m=metrics(T.obs,p); m['hybrid']=n; m['mode']=mode; rows.append(m)
WM=pd.DataFrame(rows); WM.to_csv('weightmodes.csv',index=False); print(WM.pivot(index='hybrid',columns='mode',values='R2').round(3))
print(WM.pivot(index='hybrid',columns='mode',values='RMSE').round(1))
json.dump({'|'.join(k):v for k,v in WT.items()},open('weights.json','w'),indent=0)
# district RMSE
DR=pd.DataFrame({c:P.groupby(level=0).apply(lambda g: rmse(g[c]-g.obs)) for c in P.columns[1:]}).T; DR.to_csv('district_rmse.csv'); print(DR.round(0).to_string())
DV=pd.DataFrame({c:PV.groupby(level=0).apply(lambda g: rmse(g[c]-g.obs)) for c in PV.columns[1:]}).T; DV.to_csv('district_prermse.csv'); print(DV.round(0).to_string())
# bootstrap by year
rng=np.random.default_rng(0); yrs=np.arange(2021,2026); bs={c:[] for c in ['XGB_R2','XGB_RMSE','Trend_RMSE','XGB_skill','dRMSE','TCN+XGB_RMSE']}
Pr=P.reset_index()
for _ in range(5000):
    yy=rng.choice(yrs,5); q=pd.concat([Pr[Pr.Year==y] for y in yy])
    e=q.XGB-q.obs; et=q['Trend only (Theil–Sen)']-q.obs
    bs['XGB_R2'].append(1-(e**2).sum()/((q.obs-q.obs.mean())**2).sum()); bs['XGB_RMSE'].append(rmse(e)); bs['Trend_RMSE'].append(rmse(et))
    bs['XGB_skill'].append(1-(e**2).mean()/(et**2).mean()); bs['dRMSE'].append(rmse(et)-rmse(e)); bs['TCN+XGB_RMSE'].append(rmse(q['TCN+XGB']-q.obs))
BS={k:dict(lo=float(np.percentile(v,2.5)),hi=float(np.percentile(v,97.5)),p_pos=float(np.mean(np.array(v)>0))) for k,v in bs.items()}
print(json.dumps(BS,indent=0)); json.dump(BS,open('bootstrap.json','w'))
# DM tests: XGB vs all
def dm(e1,e2):
    d=e1**2-e2**2; n=len(d); s=d.mean()/np.sqrt(d.var(ddof=0)/n); s*=np.sqrt((n-1)/n); p=2*stats.t.sf(abs(s),n-1); return s,p
rows=[]
for c in P.columns[1:]:
    if c=='XGB': continue
    s,p=dm((P.XGB-P.obs).values,(P[c]-P.obs).values); rows.append(dict(model=c,DM=s,p=p))
DM=pd.DataFrame(rows); o=np.argsort(DM.p.values); q=np.empty(len(o)); m=len(o)
ps_=DM.p.values[o]*m/np.arange(1,m+1); ps_=np.minimum.accumulate(ps_[::-1])[::-1]; q[o]=np.minimum(ps_,1); DM['q']=q
DM.to_csv('dm.csv',index=False); print(DM.round(3).to_string())
