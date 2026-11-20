import json, numpy as np, pandas as pd
from pipe import rmse, metrics
from combine import wide
from run import xgb_run
R=pd.read_csv('rolling_test_preds.csv')
o={m:rmse(R[m]-R.obs) for m in ['XGB','LSTM','TCN','XGB+LSTM','TCN+XGB','Trend only (Theil–Sen)']}
yr={y:(rmse(g.XGB-g.obs),rmse(g['Trend only (Theil–Sen)']-g.obs)) for y,g in R.groupby('Year')}
better=[y for y,(a,b) in yr.items() if a<b-2]; equal=[y for y,(a,b) in yr.items() if abs(a-b)<=2]; worse=[y for y,(a,b) in yr.items() if a>b+2]
words={0:'none',1:'one',2:'two',3:'three',4:'four',5:'all'}
order=sorted(o,key=o.get); lab={'XGB':'XGBoost','Trend only (Theil–Sen)':'the trend-only forecast'}
def L(ys): return ', '.join(str(y) for y in ys[:-1])+(' and ' if len(ys)>1 else '')+str(ys[-1]) if ys else ''
parts=[f"{lab.get(m,m)} ({o[m]:.0f} kg/ha)" for m in order]
txt=("Overall RMSE, from lowest to highest, was: "+'; '.join(parts)+". "
     f"XGBoost was better than the trend-only forecast in {words[len(better)]} of the five years"+(f" ({L(better)})" if better else "")
     +(f", about equal in {L(equal)}" if equal else "")+(f", and worse in {L(worse)}" if worse else "")+".")
best=order[0]
disc=("In the rolling-origin evaluation, "+("XGBoost also had the lowest error" if best=='XGB' else f"{lab.get(best,best)} had the lowest error ({o[best]:.0f} against {o['XGB']:.0f} kg/ha for XGBoost)")
      +", so the ranking among the best candidates is not stable and the advantage of XGBoost over the best hybrid is small; its advantage over the trend-only forecast is clearer.")
def st(n):
    W=wide(pd.read_csv(f'out_{n}.csv')).loc['test']; return {m:metrics(W.obs,W[m])['R2'] for m in ['XGB','LSTM','TCN']}
nd=st('fx_nodet5'); ols=st('fx_ols5'); cur=st('fx_curr5')
oo,_,_=xgb_run({'trend':'ols','trend_first':2011},2018,[2019,2020],2020,list(range(2021,2026)),json.load(open('xgbp_ts_prev20.json')))
t=oo[oo.stage=='test']; rec=metrics(t.obs,t.pred)['R2']
f=lambda v: f'{v:.2f}'.replace('-','−')
out=dict(RO_TEXT=txt,RO_DISC=disc,V_ND=f"{f(min(nd.values()))} to {f(max(nd.values()))}",V_OLS_X=f(ols['XGB']),
 OLS_X=f(ols['XGB']),OLS_L=f(ols['LSTM']),OLS_T=f(ols['TCN']),REC_X=f(rec),ND_X=f(nd['XGB']),ND_L=f(nd['LSTM']),ND_T=f(nd['TCN']),CUR_L=f(cur['LSTM']),CUR_T=f(cur['TCN']),
 RO=o, YR=yr)
json.dump(out,open('rolling_test.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
