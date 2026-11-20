# Machine Learning and Deep Learning Models for District-Scale Wheat Yield Prediction — Multan Division, Pakistan

Code and data accompanying the manuscript *"Evaluating machine learning and deep
learning models for district-scale wheat yield prediction: a case study of Multan
Division, Pakistan"* (Theoretical and Applied Climatology, under revision).

The repository predicts district-level winter-wheat yield for four districts of
Multan Division (Multan, Khanewal, Vehari, Lodhran) from Earth observation and
climate predictors, 2001–2025.

> **Two versions are kept in this repository.**
> - **`revision_code/code/` — revised analysis (use this).** Reproduces every number in
>   the revised manuscript under a clean protocol: train 2002–2018, validation 2019–2020,
>   test 2021–2025 used once; per-district Theil–Sen trend; models selected on 2014–2020
>   rolling forecasts. **XGBoost is the operational model.**
> - **`wheat-yield-prediction/` — original submission (kept for the record).** The SERWI
>   hybrid notebooks and their result tables. The test scores reported in the original
>   submission came from test-set tuning and are **superseded** by the revised analysis.

---

## Repository structure

```
wheat-yield-prediction-multan/
├── README.md
├── CITATION.cff
├── revision_code/
│   └── code/           revised analysis: Python scripts, inputs, outputs, and run-order READMEs
└── wheat-yield-prediction/
    ├── README.md
    ├── requirements.txt
    ├── LICENSE
    ├── data_dictionary.md
    ├── Gee_scripts/     Google Earth Engine extraction scripts (JavaScript)
    ├── Notebooks/       Google Colab notebooks (original pipeline)
    ├── Data/            input dataset, future features, and result tables (CSV)
    └── models/          trained models, scalers, trend models, and .npy arrays
```

A full description of every file and column is in **`wheat-yield-prediction/data_dictionary.md`**
(Section 7 covers the revision files).

---

## Revised analysis (`revision_code/code/`)

Run from inside `revision_code/code/`, following the run order in `README.md`,
`README_round2.md` and `README_round3.md` in that folder. Main outputs:

- XGBoost test-set results (2021–2025) against the trend-only forecast, with bootstrap
  intervals and Diebold–Mariano tests (Benjamini–Hochberg corrected).
- LSTM and TCN stability across 20 random seeds, and hybrid comparisons.
- `roll_baselines.py` — one-year-ahead rolling forecasts for 2014–2025 for XGBoost and every
  reference baseline (trend-only, OLS trend, ridge, elastic net, random forest, SVR, AR(1),
  persistence, climatology). Outputs `roll_baselines.csv` and `roll_baselines_table.csv`.

NDVI in the revised analysis is computed over a **fixed wheat mask** built only from the
pre-test years (2001–2020); see `wheat-yield-prediction/Gee_scripts/03_GEE_wheat_mask_NDVI_2001-2025.txt`.

**Reproducibility.** The XGBoost, baseline, bootstrap, Diebold–Mariano and SHAP numbers are
deterministic. The LSTM and TCN numbers were reproduced with TensorFlow 2.21 on CPU; other
TensorFlow versions or a GPU may give slightly different single-seed values.

---

## Original submission (`wheat-yield-prediction/`)

Three standalone models (XGBoost, LSTM, TCN) combined into four hybrid ensembles with the
**SERWI** (Separate Evaluation of Regression models with Weighted Integration) inverse-RMSE
weighting framework, plus 2026–2030 projections under two CMIP6 SSP2-4.5 scenarios.

### `Gee_scripts/`
- `01_GEE_historical_features_2001-2025.txt` — extracts NDVI, LST, rainfall, temperature, dewpoint (for VPD), and soil moisture, 2001–2025.
- `02_GEE_CMIP6_future_features_2026-2030.txt` — extracts CMIP6 SSP2-4.5 Tmax/heat-stress, Tmin/cold-stress, rainfall, and VPD, 2026–2030 (5-model median: ACCESS-CM2, MPI-ESM1-2-HR, MIROC6, INM-CM5-0, NorESM2-MM).
- `03_GEE_wheat_mask_NDVI_2001-2025.txt` — **(revision)** yearly and fixed wheat masks from MODIS NDVI phenology, and Mar–Apr NDVI over each mask, 2001–2025.

> CMIP6 supplies near-surface **air** temperature, whereas the historical thermal predictor is MODIS MOD11A2 **land-surface** temperature. Projected `LST_mean_C` is therefore not taken directly from CMIP6: it is constructed on the MODIS scale inside `Wheat_Future_Forecast_FINAL.ipynb` as the 2001–2020 district LST climatology plus the SSP2-4.5 temperature anomaly (division-mean ΔT = +0.48 °C over 2026–2030).

> These are JavaScript files saved as `.txt`; paste them into the Google Earth Engine Code Editor to run.

### `Data/`
- `Merged_Wheat_Multan_Final.csv` — master training dataset (100 rows = 4 districts × 25 years).
- `Future_Features_ScenarioA_Primary_2026_2030.csv`, `Future_Features_ScenarioB_Reference_2026_2030.csv` — projection inputs (20 rows each).
- `XGBoost_Results_Table.csv`, `LSTM_Results_Table.csv`, `TCN_Results_Table.csv` — standalone test-set predictions.
- `SERWI_XGB_LSTM_Results_Table.csv`, `SERWI_TCN_XGB_Results_Table.csv`, `SERWI_LSTM_TCN_Results_Table.csv`, `SERWI_Triple_Results_Table.csv` — hybrid test-set predictions.
- `Forecast_ScenarioA_Primary.csv`, `Forecast_ScenarioB_Reference.csv`, `Forecast_ALL_Scenarios.csv`, `Full_Timeline_2001_2030_ScenarioA.csv` — 2026–2030 projections.
- `Forecast_FrozenClimate_2001_2020.csv` — frozen-climate counterfactual run used to isolate the attributable climate effect.
- `Forecast_Uncertainty_Table.csv` — per-district Scenario A vs B spread.
- `Forecast_SERWI_Weights.csv` — per-district SERWI weights used for forecasting.
- `DM_Test_Results.csv` — Diebold–Mariano pairwise significance tests.
- `DM_Test_Results_HLN_BH.csv` — the same tests with the Harvey–Leybourne–Newbold small-sample correction and Benjamini–Hochberg FDR adjustment.

### `Notebooks/`
- `XGBOOST_DRIVE.ipynb`, `LSTM_DRIVE.ipynb`, `TCN_DRIVE_FINAL.ipynb` — train the three standalone models.
- `LSTM_XGboost_SERWI.ipynb`, `LSTM_TCN_SERWI.ipynb`, `XGboost_TCN_SERWI.ipynb`, `LSTM_XGboost_TCN_SERWI.ipynb` — SERWI dual and triple hybrids.
- `Wheat_Future_Forecast_FINAL.ipynb` — 2026–2030 projections under Scenarios A and B.
- `Professional_Plots_Regeneration_FINAL.ipynb` — regenerates the original figures and the Diebold–Mariano test.

### `models/`
Trained models (`xgb_wheat_model.pkl`, `lstm_wheat_model.keras`, `tcn_wheat_model.keras`),
feature/target scalers (`*_x_scaler.pkl`, `*_y_scaler.pkl`, `xgb_feature_cols.pkl`),
per-district trend models (`*_trend_models.pkl`), and the intermediate `.npy` arrays
(predictions, targets, year/district indices, and detrending components) that the
standalone notebooks save and the SERWI notebooks consume.

### How to run the original pipeline

The notebooks were written for Google Colab with files stored in Google Drive.
Inside each notebook, paths are set at the top via variables such as `BASE`,
`DRIVE_PATH`, `MODELS_PATH`, and `FORECAST_PATH`. Download the repository, upload the
folders to Google Drive (or run locally), and edit those path variables. All
intermediate artefacts are included, so any downstream notebook can be run without
re-training the earlier stages.

**Run order (only if regenerating from scratch):**

1. **GEE** — run the scripts in `Gee_scripts/` in the Earth Engine Code Editor to export the feature CSVs.
2. **Standalones** — `XGBOOST_DRIVE`, `LSTM_DRIVE`, `TCN_DRIVE_FINAL` (save models, scalers, and `.npy` arrays into `models/`).
3. **Hybrids** — `LSTM_XGboost_SERWI`, `LSTM_TCN_SERWI`, `XGboost_TCN_SERWI`, `LSTM_XGboost_TCN_SERWI` (consume the standalone `.npy` arrays).
4. **Forecast** — `Wheat_Future_Forecast_FINAL` (projects 2026–2030 for Scenarios A and B).
5. **Figures** — `Professional_Plots_Regeneration_FINAL` (regenerates the figures and the DM test).

### Scenarios (original pipeline)

Both scenarios use **identical CMIP6 SSP2-4.5 climate forcing, including land-surface
temperature**, and differ only in the assumed NDVI trajectory:

- **Scenario A (primary)** — NDVI extrapolated from its historical trend.
- **Scenario B (NDVI sensitivity case)** — NDVI regressed on projected LST.

Because the underlying climate is the same in both runs, the difference between them
measures sensitivity to the NDVI assumption alone and is **not** an uncertainty bound.
A **frozen-climate counterfactual** (`Forecast_FrozenClimate_2001_2020.csv`) repeats the
projection with the five climate predictors held at their 2001–2020 district means, so the
difference from the SSP2-4.5 run isolates the attributable climate effect.

---

## Modelling note — detrending and reported accuracy

Wheat yield is modelled in two components: a per-district technology trend fitted on the
training years, and a climate-driven residual learned by the models. The original pipeline
uses a linear trend; the revised analysis uses a **Theil–Sen** trend. Detrending is a
**training device only** — the district trend is added back to the model output before any
metric is computed, so all reported R², RMSE and MAE values refer to **observed yield**, not
to the residual.

---

## Environment

Original notebooks: see `wheat-yield-prediction/requirements.txt` (Google Colab, Python 3.10+).
`scikit-learn` is pinned to `1.6.1` because the saved `*_trend_models.pkl` were serialised
with that version.

```bash
pip install -r wheat-yield-prediction/requirements.txt
```

---

## Data sources

- **MODIS** MOD13Q1 (NDVI) and MOD11A2 (LST) — NASA LP DAAC via Google Earth Engine.
- **ERA5-Land** (rainfall, temperature, dewpoint, soil moisture) — via Google Earth Engine.
- **NASA GDDP-CMIP6**, SSP2-4.5, 5-model median — future climate projections (original pipeline).
- **District-level wheat yield** — Punjab Bureau of Statistics (PBS) and Crop Reporting Service (CRS), Government of the Punjab.

Yield is provided in maunds/acre (local unit); the manuscript reports errors in
kg/ha (1 maund/acre ≈ 98.8 kg/ha). See `data_dictionary.md`.

---

## License

Released under the MIT License (see `wheat-yield-prediction/LICENSE`). You may alternatively
wish to apply **CC BY 4.0** to the data files if depositing on Zenodo/Mendeley Data.

## Citation

If you use this code or data, please cite the associated article (citation to be
added upon publication) and this repository:

> Ahmad, A. (2026). *Machine Learning and Deep Learning Models for District-Scale Wheat Yield
> Prediction in Multan Division, Pakistan.* Zenodo. https://doi.org/10.5281/zenodo.21668885

The DOI above is the concept DOI and always resolves to the latest deposited version.
