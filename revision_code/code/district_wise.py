import json, numpy as np, pandas as pd
from run import prep, stage_frames
from models import xgb_model, XCOLS
from pipe import metrics, rmse, DIST, FEAT
p=json.load(open('xgbp_ts_prev20.json')); cfg={'trend':'theilsen'}
df=prep(cfg)
def fit_pred(dev_end, years, pooled, seed=42):
    _,a=stage_frames(df,cfg,dev_end,dev_end)
    out=[]
    if pooled:
        tr=a[(a.Year>=2002)&(a.Year<=dev_end)]; m=xgb_model(p,seed).fit(tr[XCOLS],tr.res)
        s=a[a.Year.isin(years)]; out.append(pd.DataFrame({'District':s.District,'Year':s.Year,'obs':s.y,'pred':m.predict(s[XCOLS])+s.trend}))
    else:
        for d in DIST:
            ad=a[a.District==d]; tr=ad[(ad.Year>=2002)&(ad.Year<=dev_end)]; m=xgb_model(p,seed).fit(tr[FEAT],tr.res)
            s=ad[ad.Year.isin(years)]; out.append(pd.DataFrame({'District':s.District,'Year':s.Year,'obs':s.y,'pred':m.predict(s[FEAT])+s.trend}))
    return pd.concat(out)
res={}
for pooled in [True,False]:
    k='pooled' if pooled else 'district'
    T=pd.concat([fit_pred(2020,range(2021,2026),pooled,s).assign(seed=s) for s in range(1,21)]).groupby(['District','Year']).agg(obs=('obs','first'),pred=('pred','mean'))
    T42=fit_pred(2020,range(2021,2026),pooled)
    R=pd.concat([fit_pred(Y-1,[Y],pooled) for Y in range(2014,2021)])
    res[k]=dict(test42=metrics(T42.obs,T42.pred),test20=metrics(T.obs,T.pred),pre=rmse(R.pred-R.obs),
                dist={d:rmse(g.pred-g.obs) for d,g in T42.groupby('District')})
print(json.dumps(res,indent=1,default=float)); json.dump(res,open('district_wise.json','w'),default=float)
