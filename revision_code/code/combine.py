import numpy as np, pandas as pd
from pipe import metrics, rmse, DIST
PAIRS = {'XGB+LSTM': ['XGB', 'LSTM'], 'TCN+XGB': ['TCN', 'XGB'], 'LSTM+TCN': ['LSTM', 'TCN'], 'XGB+LSTM+TCN': ['XGB', 'LSTM', 'TCN']}

def wide(out):
    """seed-averaged predictions: index (stage, District, Year) -> columns models + obs"""
    g = out.groupby(['stage', 'District', 'Year', 'model']).pred.mean().unstack('model')
    obs = out.groupby(['stage', 'District', 'Year']).obs.first()
    g['obs'] = obs; return g

def per_seed(out, seed):
    o = out[(out.seed == seed) | (out.model == 'XGB')]
    return wide(o)

def serwi(W, mode='district', bias=False):
    """W: wide frame with stage val/test. returns dict of hybrid test predictions (+val) and weights"""
    res, wts = {}, {}
    val = W.loc['val']; tst = W.loc['test']
    for name, ms in PAIRS.items():
        pv = pd.Series(0.0, index=val.index); pt = pd.Series(0.0, index=tst.index)
        for d in DIST:
            if mode == 'district':
                inv = np.array([1 / rmse(val.loc[d][m] - val.loc[d].obs) for m in ms])
            elif mode == 'global':
                inv = np.array([1 / rmse(val[m] - val.obs) for m in ms])
            else:
                inv = np.ones(len(ms))
            w = inv / inv.sum(); wts[(name, d)] = dict(zip(ms, w))
            pv.loc[d] = (val.loc[d][ms].values * w).sum(1); pt.loc[d] = (tst.loc[d][ms].values * w).sum(1)
            if bias:
                b = (val.loc[d].obs - pv.loc[d]).mean(); pt.loc[d] = pt.loc[d].values + b
        res[name] = (pv, pt)
    return res, wts

def table(W, mode='district', bias=False):
    res, wts = serwi(W, mode, bias)
    rows = []
    tst = W.loc['test']; val = W.loc['val']
    for m in ['XGB', 'LSTM', 'TCN']:
        r = metrics(tst.obs, tst[m]); r['val_RMSE'] = rmse(val[m] - val.obs); r['model'] = m; rows.append(r)
    for n, (pv, pt) in res.items():
        r = metrics(tst.obs, pt); r['val_RMSE'] = rmse(pv - val.obs); r['model'] = n; rows.append(r)
    return pd.DataFrame(rows).set_index('model'), res, wts
