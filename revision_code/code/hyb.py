import sys, numpy as np, pandas as pd
from pipe import rmse, DIST, metrics
HY={'XGB+LSTM':['XGB','LSTM'],'TCN+XGB':['TCN','XGB'],'LSTM+TCN':['LSTM','TCN'],'XGB+LSTM+TCN':['XGB','LSTM','TCN']}
def rollwide(f):
    R=pd.read_csv(f).groupby(['District','Year','model']).agg(pred=('pred','mean'),obs=('obs','first')).reset_index()
    V=R.pivot_table(index=['District','Year'],columns='model',values='pred'); V['obs']=R.groupby(['District','Year']).obs.first(); return V
def combo(F,ms,V):
    p=pd.Series(0.0,index=F.index); W={}
    for d in DIST:
        g=V.loc[d]; inv=np.array([1/rmse(g[m]-g.obs) for m in ms]); w=inv/inv.sum(); W[d]=w.round(2)
        p.loc[d]=(F.loc[d][ms].values*w).sum(1)
    return p,W
if __name__=='__main__':
    V=rollwide(sys.argv[1])
    for m in ['XGB','LSTM','TCN']: print(m,'pre',round(rmse(V[m]-V.obs),1))
    for n,ms in HY.items(): p,W=combo(V,ms,V); print(n,'pre',round(rmse(p-V.obs),1),W)
