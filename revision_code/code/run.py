import sys, json, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
from models import *

def prep(cfg):
    df = base_df(); df['dcode'] = df.District.map({d: i for i, d in enumerate(DIST)})
    return df

def stage_frames(df, cfg, dev_end, fin_end):
    kind = cfg.get('trend', 'ols'); fy = cfg.get('trend_first')
    det = cfg.get('detrend', True)
    trd = fit_trend(df, dev_end, kind, (dev_end - (2020 - fy)) if fy else None)
    trf = fit_trend(df, fin_end, kind, (fin_end - (2020 - fy)) if fy else None)
    return apply_trend(df, trd, det), apply_trend(df, trf, det)

def nn_task(args):
    kind, seed, cfg, dev_end, val_years, fin_end, test_years = args
    df = prep(cfg); a_dev, a_fin = stage_frames(df, cfg, dev_end, fin_end)
    al = cfg.get('align', 'prev'); t0 = 2005 if al == 'prev' else 2004
    tg = [(d, t) for d in DIST for t in range(t0, dev_end + 1)]
    vg = [(d, t) for d in DIST for t in val_years]
    Xtr, ytr, _ = make_seq(a_dev, tg, al); Xv, yv, mv = make_seq(a_dev, vg, al)
    pv, best, _, _ = fit_nn(kind, Xtr, ytr, seed, Xv, yv)
    mv['pred'] = pv(Xv) + mv.trend; mv['stage'] = 'val'
    tg2 = [(d, t) for d in DIST for t in range(t0, fin_end + 1)]
    sg = [(d, t) for d in DIST for t in test_years]
    Xtr2, ytr2, _ = make_seq(a_fin, tg2, al); Xs, ys, ms = make_seq(a_fin, sg, al)
    pt, _, _, _ = fit_nn(kind, Xtr2, ytr2, seed, epochs=best)
    ms['pred'] = pt(Xs) + ms.trend; ms['stage'] = 'test'
    out = pd.concat([mv, ms]); out['model'] = kind.upper(); out['seed'] = seed; out['best_epoch'] = best
    return out

def xgb_run(cfg, dev_end, val_years, fin_end, test_years, params=None):
    df = prep(cfg); a_dev, a_fin = stage_frames(df, cfg, dev_end, fin_end)
    p = params or tune_xgb(a_dev, dev_end)
    tr = a_dev[(a_dev.Year >= 2002) & (a_dev.Year <= dev_end)]; v = a_dev[a_dev.Year.isin(val_years)]
    m = xgb_model(p).fit(tr[XCOLS], tr.res)
    mv = v[['District', 'Year', 'y', 'trend']].rename(columns={'y': 'obs'}).copy(); mv['pred'] = m.predict(v[XCOLS]) + v.trend; mv['stage'] = 'val'
    tr2 = a_fin[(a_fin.Year >= 2002) & (a_fin.Year <= fin_end)]; s = a_fin[a_fin.Year.isin(test_years)]
    m2 = xgb_model(p).fit(tr2[XCOLS], tr2.res)
    ms = s[['District', 'Year', 'y', 'trend']].rename(columns={'y': 'obs'}).copy(); ms['pred'] = m2.predict(s[XCOLS]) + s.trend; ms['stage'] = 'test'
    out = pd.concat([mv, ms]); out['model'] = 'XGB'; out['seed'] = 42; out['best_epoch'] = 0
    return out, p, m2

def run(cfg, seeds, dev_end=2018, val_years=(2019, 2020), fin_end=2020, test_years=range(2021, 2026), xgb_params=None, procs=2):
    tasks = [(k, s, cfg, dev_end, list(val_years), fin_end, list(test_years)) for s in seeds for k in ('tcn', 'lstm')]
    with Pool(procs) as pool:
        res = pool.map(nn_task, tasks)
    xo, p, _ = xgb_run(cfg, dev_end, list(val_years), fin_end, list(test_years), xgb_params)
    return pd.concat(res + [xo]), p

if __name__ == '__main__':
    name = sys.argv[1]; cfg = json.loads(sys.argv[2]); n = int(sys.argv[3])
    out, p = run(cfg, list(range(1, n + 1)))
    out.to_csv(f'out_{name}.csv', index=False); json.dump(p, open(f'xgbp_{name}.json', 'w'))
    print(name, 'done', p)
