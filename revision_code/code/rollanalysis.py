import json, numpy as np, pandas as pd
from pipe import rmse, base_df, fit_trend, DIST
from final import rollwide, combo, RV
RT=rollwide('roll_ts_prev_test.csv')
df=base_df(); tr=[]
for (d,y) in RT.index:
    t=fit_trend(df,y-1,'theilsen'); tr.append(t[d].predict([[y]])[0])
RT['Trend only (Theil–Sen)']=tr
for n,ms in {'XGB+LSTM':['XGB','LSTM'],'TCN+XGB':['TCN','XGB']}.items(): RT[n]=combo(RT,ms)
R=RT.reset_index(); R.to_csv('rolling_test_preds.csv',index=False)
o={m:round(rmse(R[m]-R.obs),1) for m in ['XGB','LSTM','TCN','XGB+LSTM','TCN+XGB','Trend only (Theil–Sen)']}; print(o)
yr=[(y,rmse(g.XGB-g.obs),rmse(g['Trend only (Theil–Sen)']-g.obs)) for y,g in R.groupby('Year')]; print(yr)
win=sum(a<b for _,a,b in yr)
words={0:'none',1:'one',2:'two',3:'three',4:'four',5:'all'}
json.dump(dict(RO_XGB=f"{o['XGB']:.0f}",RO_TR=f"{o['Trend only (Theil–Sen)']:.0f}",RO_WIN=words[win]),open('rolling_test.json','w'))
