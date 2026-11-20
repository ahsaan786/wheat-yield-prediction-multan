import os, numpy as np, pandas as pd, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.preprocessing import MinMaxScaler
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data') + os.sep
K = 40 / 0.4047
DIST = ['Khanewal', 'Lodhran', 'Multan', 'Vehari']
FEAT = ['NDVI_mean', 'LST_mean_C', 'Rain_log', 'VPD_kPa', 'HeatStress_log', 'ColdStress_log',
        'Yield_lag1', 'NDVI_x_VPD', 'LST_x_Heat']

def base_df():
    df = pd.read_csv(D + 'Merged_Wheat_Multan_Final.csv').sort_values(['District', 'Year']).reset_index(drop=True)
    import os
    src = os.environ.get('NDVI_SRC', 'fixed')
    if src in ('wheat', 'fixed'):
        fn, col = {'wheat': ('NDVI_WheatMask.csv', 'NDVI_wheat'), 'fixed': ('NDVI_FixedMask.csv', 'NDVI_fixed')}[src]
        w = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn)); w['Year'] = w.Year.astype(int)
        df = df.drop(columns='NDVI_mean').merge(w[['Year', 'District', col]].rename(columns={col: 'NDVI_mean'}),
                                                on=['Year', 'District']).sort_values(['District', 'Year']).reset_index(drop=True)
    df['y'] = df.Yield_maund_acre * K
    df['lag_raw'] = df.groupby('District')['y'].shift(1)
    df['NDVI_x_VPD'] = df.NDVI_mean * df.VPD_kPa
    df['LST_x_Heat'] = df.LST_mean_C * df.HeatStress_log
    return df

def fit_trend(df, last_year, kind='ols', first_year=None):
    """per-district trend fitted on years <= last_year (and >= first_year)."""
    tr = {}
    for d in DIST:
        g = df[(df.District == d) & (df.Year <= last_year)]
        if first_year: g = g[g.Year >= first_year]
        X = g.Year.values.reshape(-1, 1); y = g.y.values
        m = TheilSenRegressor(random_state=0).fit(X, y) if kind == 'theilsen' else LinearRegression().fit(X, y)
        tr[d] = m
    return tr

def apply_trend(df, tr, detrend=True):
    out = df.copy()
    if not detrend:
        out['trend'] = 0.0; out['res'] = out.y; out['Yield_lag1'] = out.lag_raw
        return out
    out['trend'] = [tr[d].predict([[y]])[0] for d, y in zip(out.District, out.Year)]
    out['trend_prev'] = [tr[d].predict([[y - 1]])[0] for d, y in zip(out.District, out.Year)]
    out['res'] = out.y - out.trend
    out['Yield_lag1'] = out.lag_raw - out.trend_prev     # detrended lag for all models
    return out

def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
def metrics(obs, pred):
    obs = np.asarray(obs); e = np.asarray(pred) - obs
    return dict(R2=1 - (e ** 2).sum() / ((obs - obs.mean()) ** 2).sum(), RMSE=rmse(e), MAE=float(np.abs(e).mean()), Bias=float(e.mean()))
