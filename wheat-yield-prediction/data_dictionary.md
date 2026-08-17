# Data Dictionary

This document describes every data file in the repository and the meaning and
units of each column. All data cover four districts of **Multan Division,
Punjab, Pakistan** — Multan, Khanewal, Vehari, and Lodhran.

---

## 1. `Merged_Wheat_Multan_Final.csv` — master training dataset
100 rows = 4 districts × 25 years (2001–2025). One row per district-year.

| Column | Description | Unit |
|---|---|---|
| `Year` | Wheat season (harvest year) | year |
| `District` | District name | — |
| `Yield_maund_acre` | Observed wheat yield (target variable) | maunds/acre |
| `NDVI_mean` | Mean NDVI, Mar–Apr (grain filling), MODIS MOD13Q1, cropland-masked | index (0–1) |
| `LST_mean_C` | Mean land-surface temperature, Nov–Apr, MODIS MOD11A2, QC-filtered | °C |
| `Rain_log` | log1p of seasonal rainfall (Nov–Apr), CHIRPS/ERA5-Land | log(mm) |
| `VPD_kPa` | Vapour-pressure deficit, Mar–Apr, ERA5-Land | kPa |
| `HeatStress_log` | log1p of heat-stress days (Tmax > 34 °C, Mar–Apr) | log(days) |
| `ColdStress_log` | log1p of cold-stress days (Tmin < 5 °C, Nov–Feb) | log(days) |

> **Yield unit note.** `Yield_maund_acre` is in the local unit *maunds per acre*
> (1 maund = 40 kg; 1 acre = 0.4047 ha, so **1 maund/acre ≈ 98.8 kg/ha**).
> The manuscript reports errors (RMSE, MAE) in **kg/ha**; the conversion and the
> two-component linear detrending are applied inside the notebooks.
>
> **Detrending note.** Detrending is a *training device only*. Models are fitted on
> the climate residual, but the per-district linear trend is added back to the model
> output before any metric is computed, so the reported **R², RMSE and MAE refer to
> observed yield**, not to the residual. The interaction terms `NDVI×VPD` and
> `LST×HeatStress` are derived inside the notebooks from the columns above and are
> not stored separately.

---

## 2. Future feature files (projection inputs, 2026–2030)
`Future_Features_ScenarioA_Primary_2026_2030.csv` and
`Future_Features_ScenarioB_Reference_2026_2030.csv` — 20 rows each
(4 districts × 5 years). Same predictor columns as the training set (no yield),
derived from NASA GDDP-CMIP6, SSP2-4.5, pixel-wise median of five GCMs
(ACCESS-CM2, MPI-ESM1-2-HR, MIROC6, INM-CM5-0, NorESM2-MM).

| Column | Description | Unit |
|---|---|---|
| `Year` | Projection year (2026–2030) | year |
| `District` | District name | — |
| `NDVI_mean`, `LST_mean_C`, `Rain_log`, `VPD_kPa`, `HeatStress_log`, `ColdStress_log` | As defined above | as above |

- **Scenario A (Primary):** CMIP6 SSP2-4.5 climate with NDVI-trend extrapolation.
- **Scenario B (Reference):** SSP2-4.5 with climate-driven NDVI regression (conservative stress bound).

For the projection period the district technology trend is held fixed at its 2025
value rather than extrapolated forward, so projected variation reflects the modelled
climate response alone.

---

## 3. Model result tables
Standalone: `XGBoost_Results_Table.csv`, `LSTM_Results_Table.csv`, `TCN_Results_Table.csv`
Hybrids: `SERWI_XGB_LSTM_Results_Table.csv`, `SERWI_TCN_XGB_Results_Table.csv`,
`SERWI_LSTM_TCN_Results_Table.csv`, `SERWI_Triple_Results_Table.csv`
Test period 2021–2025 (20 rows = 4 districts × 5 years).

| Column | Description | Unit |
|---|---|---|
| `Year` | Test year | year |
| `District` | District name | — |
| `Actual` | Observed yield | maunds/acre |
| `Predicted` | Model-predicted yield (trend restored) | maunds/acre |
| `Error` / `Residual` | Actual − Predicted | maunds/acre |

> `Predicted` already includes the re-added district trend, so `Actual` and
> `Predicted` are directly comparable observed-yield quantities.

---

## 4. Forecast tables
`Forecast_ScenarioA_Primary.csv`, `Forecast_ScenarioB_Reference.csv`,
`Forecast_ALL_Scenarios.csv`

| Column | Description | Unit |
|---|---|---|
| `Year` | 2026–2030 | year |
| `District` | District name | — |
| `XGB`, `LSTM`, `TCN` | Standalone projected yield | maunds/acre |
| `SERWI_XGB_LSTM`, `SERWI_TCN_XGB`, `SERWI_LSTM_TCN`, `SERWI_Triple` | Hybrid projected yield | maunds/acre |
| `Scenario` | `ScenarioA_Primary` or `ScenarioB_Reference` | — |

`Full_Timeline_2001_2030_ScenarioA.csv` — combined historical (2001–2025) +
projected (2026–2030) series per district, used for the timeline figures.

| Column | Description | Unit |
|---|---|---|
| `Year` | 2001–2030 | year |
| `District` | District name | — |
| `Yield_kg_ha` | Observed (2001–2025) or projected (2026–2030) yield | kg/ha |
| `Scenario` | `Historical` or scenario label | — |

---

## 5. `DM_Test_Results.csv` — Diebold–Mariano pairwise tests

| Column | Description |
|---|---|
| `Benchmark` | Reference model |
| `Challenger` | Compared model |
| `DM_Statistic` | Diebold–Mariano test statistic |
| `p_value` | Two-sided p-value |
| `Significant` | Yes/No at α = 0.05 |
| `Better_Model` | Model with lower forecast error |

---

## 6. Intermediate arrays and model objects
- `*_y_pred_*`, `*_y_test`, `*_y_train`, `*_yr_test`, `*_dist_test`, `*_trend_*` (`.npy`)
  — per-model predictions, targets, year/district indices, and detrending components
  saved by the standalone notebooks and consumed by the SERWI hybrid notebooks.
- `xgb_wheat_model.pkl`, `lstm_wheat_model.keras`, `tcn_wheat_model.keras` — trained models.
- `*_x_scaler.pkl`, `*_y_scaler.pkl`, `xgb_feature_cols.pkl` — feature/target scalers and column order.
- `*_trend_models.pkl` — per-district linear trend models (for detrending / re-adding trend).
