import json, itertools, numpy as np, pandas as pd
from pipe import *
from xgbcore import XGB_GRID, xgb_model, XCOLS
W=['Rain_log','VPD_kPa','HeatStress_log','ColdStress_log','Yield_lag1','dcode']
def prep():
    df=base_df(); df['dcode']=df.District.map({d:i for i,d in enumerate(DIST)}); return df
df=prep()
def frames(dev_end,fin_end): return apply_trend(df,fit_trend(df,dev_end,'theilsen')), apply_trend(df,fit_trend(df,fin_end,'theilsen'))
def tune(a,train_end,cols,folds=5):
    best,bp=1e18,None; keys=list(XGB_GRID); tr=a[(a.Year>=2002)&(a.Year<=train_end)]
    for c in itertools.product(*[XGB_GRID[k] for k in keys]):
        p=dict(zip(keys,c)); se=0
        for vy in range(train_end-folds+1,train_end+1):
            A=tr[tr.Year<vy]; B=tr[tr.Year==vy]; se+=((xgb_model(p).fit(A[cols],A.res).predict(B[cols])-B.res)**2).sum()
        if se<best: best,bp=se,p
    return bp
P0=json.load(open('xgbp_main_re.json'))
out={}
for name,cols in [('full',XCOLS),('weather',W)]:
    for mode in ['retuned','fixed']:
        e=[]
        for Y in range(2014,2021):
            ad,af=frames(Y-3,Y-1)
            p=tune(ad,Y-3,cols) if mode=='retuned' else P0
            tr=af[(af.Year>=2002)&(af.Year<=Y-1)]; s=af[af.Year==Y]
            e.append(xgb_model(p).fit(tr[cols],tr.res).predict(s[cols])+s.trend-s.y)
        out[(name,mode,'pre')]=rmse(pd.concat(e))
    # test with params tuned on 2018 for these cols
    ad,af=frames(2018,2020); p=tune(ad,2018,cols); tr=af[(af.Year>=2002)&(af.Year<=2020)]; s=af[af.Year>=2021]
    out[(name,'tuned2018','test')]=metrics(s.y,xgb_model(p).fit(tr[cols],tr.res).predict(s[cols])+s.trend)
    tr=af[(af.Year>=2002)&(af.Year<=2020)]
    out[(name,'P0','test')]=metrics(s.y,xgb_model(P0).fit(tr[cols],tr.res).predict(s[cols])+s.trend)
    print(name,{k:v for k,v in out.items() if k[0]==name},flush=True)
