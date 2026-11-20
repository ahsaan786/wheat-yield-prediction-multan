# Additional check (third-round revision)

Run from the `code` folder after the runs in README.md and README_round2.md.

`python roll_baselines.py` (about 10 minutes on 2 cores)

One-year-ahead rolling forecasts for 2014-2025 for XGBoost and every reference baseline, with the same rule as
`rolling.py` (tune on years up to Y-3, refit on years up to Y-1, forecast year Y). It gives:

1. the pre-test rolling RMSE (2014-2020) of the baselines, reported in Table 3;
2. the twelve-year comparison of XGBoost with the trend-only forecast (Section 3.5):
   RMSE, years won, and a year-block bootstrap interval of the RMSE difference;
3. the trend-choice check for XGBoost (Theil-Sen against OLS) on the rolling forecasts (Section 2.5).

Outputs: `roll_baselines.csv` (all forecasts) and `roll_baselines_table.csv` (RMSE summary).

Reproducibility note: the XGBoost, baseline, bootstrap, Diebold-Mariano and SHAP numbers are deterministic.
The LSTM and TCN numbers were reproduced exactly with TensorFlow 2.21 on CPU; other TensorFlow versions or a GPU
may give slightly different single-seed values.
