import numpy as np, pandas as pd, itertools
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from pipe import *

def Xmat(a):
    X = a[FEAT].copy()
    for d in DIST[1:]: X['d_' + d] = (a.District == d).astype(float)
    return X

def cv_select(a, make, grid, dev_end=2018, folds=5):
    best, bp = 1e18, None
    tr = a[(a.Year >= 2002) & (a.Year <= dev_end)]
    for p in grid:
        se = 0
        for vy in range(dev_end - folds + 1, dev_end + 1):
            A = tr[tr.Year < vy]; B = tr[tr.Year == vy]
            m = make(p).fit(Xmat(A), A.res); se += ((m.predict(Xmat(B)) - B.res) ** 2).sum()
        if se < best: best, bp = se, p
    return bp

MODELS = {
 'Ridge': (lambda p: make_pipeline(StandardScaler(), Ridge(alpha=p)), [0.1, 0.3, 1, 3, 10, 30, 100, 300]),
 'Elastic net': (lambda p: make_pipeline(StandardScaler(), ElasticNet(alpha=p[0], l1_ratio=p[1], max_iter=20000)),
                 list(itertools.product([0.3, 1, 3, 10, 30, 100], [0.1, 0.5, 0.9]))),
 'Random forest': (lambda p: RandomForestRegressor(n_estimators=500, max_depth=p[0], min_samples_leaf=p[1], random_state=42, n_jobs=1),
                   list(itertools.product([2, 3, 5, None], [2, 5, 10]))),
 'SVR (RBF)': (lambda p: make_pipeline(StandardScaler(), SVR(C=p[0], gamma=p[1], epsilon=10)),
               list(itertools.product([10, 100, 1000], ['scale', 0.01, 0.1]))),
}

def run_baselines(test_years=range(2021, 2026), dev_end=2018, fin_end=2020, kind0='ols'):
    df = base_df()
    a_dev = apply_trend(df, fit_trend(df, dev_end, kind0)); a = apply_trend(df, fit_trend(df, fin_end, kind0))
    tst = a[a.Year.isin(test_years)].copy(); rows = {}
    tr = a[(a.Year >= 2002) & (a.Year <= fin_end)]
    for n, (mk, grid) in MODELS.items():
        p = cv_select(a_dev, mk, grid, dev_end)
        m = mk(p).fit(Xmat(tr), tr.res); rows[n] = (m.predict(Xmat(tst)) + tst.trend.values, p)
    # AR(1) on residual (pooled)
    g = tr.dropna(subset=['Yield_lag1']); phi = (g.res * g.Yield_lag1).sum() / (g.Yield_lag1 ** 2).sum()
    rows['AR(1) residual'] = (tst.trend.values + phi * tst.Yield_lag1.values, round(phi, 3))
    rows['Persistence'] = (tst.lag_raw.values, None)
    rows['Trend only (OLS 2001–2020)'] = (tst.trend.values, None)
    for kind, fy, lab in [('ols', 2011, 'Trend only (recent window 2011–2020)'), ('theilsen', None, 'Trend only (Theil–Sen)')]:
        tr2 = fit_trend(df, fin_end, kind, fy); rows[lab] = (np.array([tr2[d].predict([[y]])[0] for d, y in zip(tst.District, tst.Year)]), None)
    clim = df[df.Year <= fin_end].groupby('District').y.mean(); rows['Climatology'] = (tst.District.map(clim).values, None)
    res = []
    for n, (p, par) in rows.items():
        m = metrics(tst.y, p); m['model'] = n; m['param'] = str(par); res.append(m)
    preds = pd.DataFrame({n: v[0] for n, v in rows.items()}); preds['District'] = tst.District.values; preds['Year'] = tst.Year.values; preds['obs'] = tst.y.values
    return pd.DataFrame(res).set_index('model'), preds

if __name__ == '__main__':
    import sys; k = sys.argv[1] if len(sys.argv) > 1 else 'ols'
    t, p = run_baselines(kind0=k); print(t.round(3)); import os; sfx = '' if os.environ.get('NDVI_SRC', 'crop') == 'crop' else '_' + os.environ['NDVI_SRC']
    p.to_csv(f'baseline_preds_{k}{sfx}.csv', index=False); t.to_csv(f'baseline_table_{k}{sfx}.csv')
