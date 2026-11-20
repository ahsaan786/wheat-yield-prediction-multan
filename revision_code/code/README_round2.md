# Additional checks (second-round revision)

Run from the `code` folder after the main runs in README.md.

1. `python run.py ts_curr20 '{"trend":"theilsen","align":"curr"}' 20` and
   `python rolling.py ts_curr '{"trend":"theilsen","align":"curr"}' 5 2014 2020`:
   matched-information check, sequences ending in the forecast season (t-2 to t).
2. `python hyb.py roll_ts_curr.csv` (and `roll_ts_prev.csv`): pre-test RMSE and district weights of all candidates.
3. `python chk13.py`: weather-only XGBoost under the same protocol as the full model
   (re-tuned at each rolling origin; also with the main hyperparameters fixed).
4. `python dm14.py`: Diebold-Mariano tests on the 20 errors and on year-mean loss differences (n = 5), with BH adjustment.

`xgbcore.py` holds the XGBoost grid and tuning without importing TensorFlow.
