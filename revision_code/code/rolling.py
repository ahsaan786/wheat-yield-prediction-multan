import sys, json, pandas as pd
from run import run
name=sys.argv[1]; cfg=json.loads(sys.argv[2]); n=int(sys.argv[3]); years=range(int(sys.argv[4]),int(sys.argv[5])+1)
outs=[]
for Y in years:
    o,p=run(cfg,list(range(1,n+1)),dev_end=Y-3,val_years=(Y-2,Y-1),fin_end=Y-1,test_years=[Y])
    o=o[o.stage=='test'].copy(); o['origin']=Y; outs.append(o); print(Y,'done',flush=True)
    pd.concat(outs).to_csv(f'roll_{name}.csv',index=False)
