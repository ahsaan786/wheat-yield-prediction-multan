# Hybrid ML–DL Ensemble for District-Scale Wheat Yield Prediction — Multan Division, Pakistan

Code and data accompanying the manuscript *"Development of a Hybrid Machine
Learning–Deep Learning Ensemble for District-Scale Wheat Yield Prediction
Using Earth Observation and Climate Data."*

The pipeline predicts district-level winter-wheat yield across four districts of
Multan Division (Multan, Khanewal, Vehari, Lodhran) for 2001–2025 and projects
yields for 2026–2030 under two CMIP6 SSP2-4.5 scenarios. Three standalone models
(XGBoost, LSTM, TCN) are combined into four hybrid ensembles using the **SERWI**
(Separate Evaluation of Regression models with Weighted Integration) inverse-RMSE
weighting framework.

---

## Repository structure

```
wheat-yield-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── data_dictionary.md
├── Gee_scripts/     Google Earth Engine extraction scripts (JavaScript)
├── Notebooks/       Google Colab notebooks (the full pipeline)
├── Data/            input dataset, future features, and result tables (CSV)
└── models/          trained models, scalers, trend models, and .npy arrays
```

A full description of every file and column is in **`data_dictionary.md`**.

### `Gee_scripts/`
- `01_GEE_historical_features_2001-2025.txt` — extracts NDVI, LST, rainfall, temperature, dewpoint (for VPD), and soil moisture, 2001–2025.
- `02_GEE_CMIP6_future_features_2026-2030.txt` — extracts CMIP6 SSP2-4.5 Tmax/heat-stress, Tmin/cold-stress, rainfall, and VPD, 2026–2030 (5-model median: ACCESS-CM2, MPI-ESM1-2-HR, MIROC6, INM-CM5-0, NorESM2-MM).

> These are JavaScript files saved as `.txt`; paste them into the Google Earth Engine Code Editor to run.

### `Data/`
- `Merged_Wheat_Multan_Final.csv` — master training dataset (100 rows = 4 districts × 25 years).
- `Future_Features_ScenarioA_Primary_2026_2030.csv`, `Future_Features_ScenarioB_Reference_2026_2030.csv` — projection inputs (20 rows each).
- `XGBoost_Results_Table.csv`, `LSTM_Results_Table.csv`, `TCN_Results_Table.csv` — standalone test-set predictions.
- `SERWI_XGB_LSTM_Results_Table.csv`, `SERWI_TCN_XGB_Results_Table.csv`, `SERWI_LSTM_TCN_Results_Table.csv`, `SERWI_Triple_Results_Table.csv` — hybrid test-set predictions.
- `Forecast_ScenarioA_Primary.csv`, `Forecast_ScenarioB_Reference.csv`, `Forecast_ALL_Scenarios.csv`, `Full_Timeline_2001_2030_ScenarioA.csv` — 2026–2030 projections.
- `DM_Test_Results.csv` — Diebold–Mariano pairwise significance tests.

### `Notebooks/`
- `XGBOOST_DRIVE.ipynb`, `LSTM_DRIVE.ipynb`, `TCN_DRIVE_FINAL.ipynb` — train the three standalone models.
- `LSTM_XGboost_SERWI.ipynb`, `LSTM_TCN_SERWI.ipynb`, `XGboost_TCN_SERWI.ipynb`, `LSTM_XGboost_TCN_SERWI.ipynb` — SERWI dual and triple hybrids.
- `Wheat_Future_Forecast_FINAL.ipynb` — 2026–2030 projections under Scenarios A and B.
- `Professional_Plots_Regeneration_FINAL.ipynb` — regenerates all manuscript figures and the Diebold–Mariano test.

### `models/`
Trained models (`xgb_wheat_model.pkl`, `lstm_wheat_model.keras`, `tcn_wheat_model.keras`),
feature/target scalers (`*_x_scaler.pkl`, `*_y_scaler.pkl`, `xgb_feature_cols.pkl`),
per-district trend models (`*_trend_models.pkl`), and the intermediate `.npy` arrays
(predictions, targets, year/district indices, and detrending components) that the
standalone notebooks save and the SERWI notebooks consume.

---

## How to run

The notebooks were written for Google Colab with files stored in Google Drive.
Inside each notebook, paths are set at the top via variables such as `BASE`,
`DRIVE_PATH`, `MODELS_PATH`, and `FORECAST_PATH`.

**To reproduce:** download this repository, upload the folders to your Google
Drive (or run locally), and edit those path variables at the top of each notebook
to point to wherever you placed the `Data/` and `models/` contents. All
intermediate artefacts are included, so any downstream notebook can be run
without re-training the earlier stages.

**Run order (only if regenerating from scratch):**

1. **GEE** — run the two scripts in `Gee_scripts/` in the Earth Engine Code Editor to export the feature CSVs.
2. **Standalones** — `XGBOOST_DRIVE`, `LSTM_DRIVE`, `TCN_DRIVE_FINAL` (save models, scalers, and `.npy` arrays into `models/`).
3. **Hybrids** — `LSTM_XGboost_SERWI`, `LSTM_TCN_SERWI`, `XGboost_TCN_SERWI`, `LSTM_XGboost_TCN_SERWI` (consume the standalone `.npy` arrays).
4. **Forecast** — `Wheat_Future_Forecast_FINAL` (projects 2026–2030 for Scenarios A and B).
5. **Figures** — `Professional_Plots_Regeneration_FINAL` (regenerates all figures and the DM test).

---

## Modelling note — detrending and reported accuracy

Wheat yield is modelled in two components: a per-district linear technology trend
fitted on the pre-test years, and a climate-driven residual learned by the models.
Detrending is a **training device only** — the district trend is added back to the
model output before any metric is computed, so all reported R², RMSE and MAE values
refer to **observed yield**, not to the residual. For the 2026–2030 projections the
trend is held fixed at its 2025 value rather than extrapolated forward, so projected
variation reflects the modelled climate response alone.

---

## Environment

See `requirements.txt`. Original runs used Google Colab (Python 3.10+).
`scikit-learn` is pinned to `1.6.1` because the saved `*_trend_models.pkl`
were serialised with that version.

```bash
pip install -r requirements.txt
```

---

## Data sources

- **MODIS** MOD13Q1 (NDVI) and MOD11A2 (LST) — NASA LP DAAC via Google Earth Engine.
- **ERA5-Land** (rainfall, temperature, dewpoint, soil moisture) and **CHIRPS** (rainfall) — via Google Earth Engine.
- **NASA GDDP-CMIP6**, SSP2-4.5, 5-model median — future climate projections.
- **District-level wheat yield** — Punjab Bureau of Statistics (PBS) and Crop Reporting Service (CRS), Government of the Punjab.

Yield is provided in maunds/acre (local unit); the manuscript reports errors in
kg/ha (1 maund/acre ≈ 98.8 kg/ha). See `data_dictionary.md`.

---

## License

Released under the MIT License (see `LICENSE`). You may alternatively wish to
apply **CC BY 4.0** to the data files if depositing on Zenodo/Mendeley Data.

## Citation

If you use this code or data, please cite the associated article (citation to be
added upon publication) and this repository.
