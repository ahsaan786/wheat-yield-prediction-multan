# Third-round check: one-year-ahead rolling forecasts 2014-2025 for XGBoost AND every baseline,
# with the same rule as rolling.py (tune on years <= Y-3, refit on years <= Y-1, forecast Y).
# Gives the pre-test RMSE of the baselines (Table 3) and the twelve-year comparison (Section 3.5).
# Run from the code folder:  python roll_baselines.py     (about 10 minutes on 2 cores)
import json, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
from pipe import *
from xgbcore import xgb_model, XCOLS, tune_xgb
import baselines as B

TS = 'Trend only (Theil–Sen)'


def one(Y):
    df = base_df(); df['dcode'] = df.District.map({d: i for i, d in enumerate(DIST)})
    ad = apply_trend(df, fit_trend(df, Y - 3, 'theilsen')); af = apply_trend(df, fit_trend(df, Y - 1, 'theilsen'))
    tr = af[(af.Year >= 2002) & (af.Year <= Y - 1)]; s = af[af.Year == Y]
    o = s[['District', 'Year', 'y']].rename(columns={'y': 'obs'}).copy()
    p = tune_xgb(ad, Y - 3)
    o['XGB'] = xgb_model(p).fit(tr[XCOLS], tr.res).predict(s[XCOLS]) + s.trend.values
    o[TS] = s.trend.values
    for kind, fy, lab in [('ols', None, 'Trend only (OLS)'), ('ols', Y - 10, 'Trend only (recent window)')]:
        t2 = fit_trend(df, Y - 1, kind, fy); o[lab] = [t2[d].predict([[Y]])[0] for d in s.District]
    for n, (mk, grid) in B.MODELS.items():
        pp = B.cv_select(ad, mk, grid, Y - 3)
        o[n] = mk(pp).fit(B.Xmat(tr), tr.res).predict(B.Xmat(s)) + s.trend.values
    g = tr.dropna(subset=['Yield_lag1']); phi = (g.res * g.Yield_lag1).sum() / (g.Yield_lag1 ** 2).sum()
    o['AR(1) residual'] = s.trend.values + phi * s.Yield_lag1.values
    o['Persistence'] = s.lag_raw.values
    o['Climatology'] = s.District.map(df[df.Year <= Y - 1].groupby('District').y.mean()).values
    # XGBoost with an OLS trend (trend-choice check, Section 2.5)
    ad2 = apply_trend(df, fit_trend(df, Y - 3, 'ols')); af2 = apply_trend(df, fit_trend(df, Y - 1, 'ols'))
    tr2 = af2[(af2.Year >= 2002) & (af2.Year <= Y - 1)]; s2 = af2[af2.Year == Y]
    o['XGB (OLS trend)'] = xgb_model(tune_xgb(ad2, Y - 3)).fit(tr2[XCOLS], tr2.res).predict(s2[XCOLS]) + s2.trend.values
    return o


if __name__ == '__main__':
    with Pool(2) as pool:
        R = pd.concat(pool.map(one, list(range(2014, 2026)), chunksize=1))
    R.to_csv('roll_baselines.csv', index=False)
    cols = [c for c in R.columns if c not in ('District', 'Year', 'obs')]
    pre, te = R[R.Year <= 2020], R[R.Year >= 2021]
    T = pd.DataFrame({'pre-test 2014-2020': {c: rmse(pre[c] - pre.obs) for c in cols},
                      'test 2021-2025 (rolling)': {c: rmse(te[c] - te.obs) for c in cols},
                      'all 2014-2025': {c: rmse(R[c] - R.obs) for c in cols}}).round(1)
    print(T.sort_values('pre-test 2014-2020').to_string())
    # twelve-year comparison of XGBoost with the Theil-Sen trend-only forecast
    d = ((R.XGB - R.obs) ** 2 - (R[TS] - R.obs) ** 2).groupby(R.Year).mean()
    print('years in which XGBoost had the lower squared error:', int((d < 0).sum()), 'of', len(d), sorted(d[d < 0].index))
    print(R.groupby('Year').apply(lambda g: pd.Series({'XGB': rmse(g.XGB - g.obs), 'Trend': rmse(g[TS] - g.obs)})).round(0).T.to_string())
    rng = np.random.default_rng(0); yrs = np.arange(2014, 2026); G = {y: R[R.Year == y] for y in yrs}; bs = []
    for _ in range(5000):
        q = pd.concat([G[y] for y in rng.choice(yrs, 12)]); bs.append(rmse(q[TS] - q.obs) - rmse(q.XGB - q.obs))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('year-block bootstrap of RMSE(trend) - RMSE(XGB), 2014-2025: 95%% interval %.0f to %.0f kg/ha, positive in %.0f%%' % (lo, hi, 100 * np.mean(np.array(bs) > 0)))
    T.to_csv('roll_baselines_table.csv')
