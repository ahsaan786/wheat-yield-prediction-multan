import json, numpy as np, pandas as pd, shap
from run import prep, stage_frames
from models import xgb_model, XCOLS
from pipe import metrics, DIST, fit_trend, K, D
import os; MAIN=os.environ.get('MAINRUN','ts_prev20'); p=json.load(open(f'xgbp_{MAIN}.json')); cfg={'trend':'theilsen'}
df=prep(cfg); _,a=stage_frames(df,cfg,2018,2020)
tr=a[(a.Year>=2002)&(a.Year<=2020)]; te=a[a.Year>=2021]
r2=[]
for s in range(1,21):
    m=xgb_model(p,seed=s).fit(tr[XCOLS],tr.res); r2.append(metrics(te.y,m.predict(te[XCOLS])+te.trend)['R2'])
print('XGB seeds R2 mean %.3f sd %.3f min %.3f max %.3f'%(np.mean(r2),np.std(r2),min(r2),max(r2)))
m=xgb_model(p).fit(tr[XCOLS],tr.res)
X=a[a.Year>=2002][XCOLS]; ex=shap.TreeExplainer(m); sv=ex.shap_values(X)
S=pd.DataFrame(sv,columns=XCOLS); S.to_csv(f'shap_xgb_{MAIN}.csv',index=False); X.reset_index(drop=True).to_csv(f'shap_X_{MAIN}.csv',index=False)
imp=S.abs().mean().sort_values(ascending=False); print(imp.round(1))
# signed association: corr of feature value with its shap
print({c:round(np.corrcoef(X[c],S[c])[0,1],2) for c in XCOLS if S[c].abs().sum()>0})
# projections
Y=a.copy(); full=fit_trend(df,2025,'theilsen')
a25=stage_frames(df,cfg,2025,2025)[1]; trn=a25[(a25.Year>=2002)]
mf=xgb_model(p).fit(trn[XCOLS],trn.res)
res=[]
DT={2026:0.42,2027:0.45,2028:0.48,2029:0.51,2030:0.54}
hb=df[df.Year<=2020].groupby('District').LST_mean_C.mean()
base=pd.read_csv(D+'Future_Features_ScenarioA_Primary_2026_2030.csv')
base['LST_mean_C']=[hb[d]+DT[y] for d,y in zip(base.District,base.Year)]
pl=np.polyfit(df.LST_mean_C,df.NDVI_mean,1); print('LST-NDVI R2 %.2f'%(np.corrcoef(df.LST_mean_C,df.NDVI_mean)[0,1]**2))
FA=base.copy(); FA['NDVI_mean']=[np.polyval(np.polyfit(df[df.District==d].Year,df[df.District==d].NDVI_mean,1),y) for d,y in zip(FA.District,FA.Year)]
FB=base.copy(); FB['NDVI_mean']=np.polyval(pl,FB.LST_mean_C)
tr0=df[df.Year<=2020]; out=0
for c in ['NDVI_mean','LST_mean_C','Rain_log','VPD_kPa','HeatStress_log','ColdStress_log']:
    for F in (FA,FB): out+=((F[c]<tr0[c].min())|(F[c]>tr0[c].max())).sum()
print('exogenous values outside 2001-2020 range (A+B, 6 vars):',out)
for sc,F in [('A',FA),('B',FB)]:
    F=F.copy(); F['NDVI_x_VPD']=F.NDVI_mean*F.VPD_kPa; F['LST_x_Heat']=F.LST_mean_C*F.HeatStress_log
    F['dcode']=F.District.map({d:i for i,d in enumerate(DIST)})
    for mode in ['frozen','continued']:
        for _,r in F.iterrows():
            d=r.District; yr=r.Year; h=df[df.District==d].set_index('Year').y
            lag=h[2025] if yr==2026 else h[[2023,2024,2025]].mean()
            ty=lambda y: full[d].predict([[min(y,2025) if mode=='frozen' else y]])[0]
            lagref=2025 if yr==2026 else 2024   # trend at the lag's reference year
            x=r.copy(); x['Yield_lag1']=lag-ty(lagref)
            pred=mf.predict(pd.DataFrame([x[XCOLS].astype(float)]))[0]+ty(yr)
            res.append(dict(Scenario=sc,trend=mode,District=d,Year=yr,pred=pred))
R=pd.DataFrame(res); R.to_csv(f'proj_xgb_{MAIN}.csv',index=False)
print(R.groupby(['Scenario','trend','District']).pred.mean().unstack().round(0))
h=df.pivot(index='Year',columns='District',values='y'); print('2021-25 mean',h.loc[2021:2025].mean().round(0).to_dict(),'2025',h.loc[2025].round(0).to_dict())
