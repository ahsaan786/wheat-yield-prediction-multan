import itertools, numpy as np, pandas as pd
from xgboost import XGBRegressor
from pipe import *
XGB_GRID = dict(n_estimators=[50, 100, 150], max_depth=[2, 3], learning_rate=[0.01, 0.05, 0.1],
                reg_alpha=[0.1, 0.5, 1.0], reg_lambda=[1.0, 3.0, 5.0], min_child_weight=[3, 5, 7])
XCOLS = FEAT + ['dcode']

def xgb_model(p, seed=42):
    return XGBRegressor(subsample=0.8, colsample_bytree=0.8, gamma=0.5, random_state=seed, verbosity=0, n_jobs=1, **p)

def tune_xgb(df, train_end, folds=5):
    """expanding time-ordered CV: validation years train_end-folds+1 .. train_end."""
    best, bp = 1e18, None
    keys = list(XGB_GRID); combos = list(itertools.product(*[XGB_GRID[k] for k in keys]))
    tr = df[(df.Year >= 2002) & (df.Year <= train_end)]
    for c in combos:
        p = dict(zip(keys, c)); se = 0
        for vy in range(train_end - folds + 1, train_end + 1):
            a = tr[tr.Year < vy]; b = tr[tr.Year == vy]
            m = xgb_model(p).fit(a[XCOLS], a.res); se += ((m.predict(b[XCOLS]) - b.res) ** 2).sum()
        if se < best: best, bp = se, p
    return bp
