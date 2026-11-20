# Revised pipeline (Reviewer 2 revision)

Data path is set in `pipe.py` (`D = .../Data/`). The NDVI input is chosen with the environment variable
`NDVI_SRC`: `fixed` (default, fixed MODIS wheat mask), `wheat` (yearly wheat mask) or `crop` (original cropland NDVI).
The mask CSVs must sit next to `pipe.py`.

1. `python run.py ts_prev20 '{"trend":"theilsen","align":"prev"}' 20`: main run, 20 seeds, test years used once
2. `python rolling.py ts_prev '{"trend":"theilsen","align":"prev"}' 5 2014 2020`: pre-test rolling forecasts (SERWI weights, model selection)
3. `python rolling.py ts_prev_test '{"trend":"theilsen","align":"prev"}' 5 2021 2025`: rolling-origin test
4. Sensitivity runs: `detrend:false`, `trend:ols`, `align:curr` (5 seeds), and NDVI_SRC=crop / wheat (20 seeds)
5. `python baselines.py theilsen` and `python baselines.py ols`: reference baselines
6. `python final.py`, `python rollanalysis.py`, `python xgb_final.py`, `python makero.py`: tables, bootstrap, DM tests, SHAP, projections
7. `python figs2.py "fig3()" "fig4()" ...` and `python fig2b.py`: figures
