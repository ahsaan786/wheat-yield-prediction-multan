# Data Dictionary

This document describes every data file in the repository and the meaning and
units of each column. All data cover four districts of **Multan Division,
Punjab, Pakistan** — Multan, Khanewal, Vehari, and Lodhran.

> **Which files go with which version of the paper.** Sections 1–6 describe the files
> from the original submission (`wheat-yield-prediction/Data/` and `models/`). The
> results reported in the **revised manuscript** are reproduced by the scripts in
> `revision_code/code/`; the files produced there are described in Section 7.

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
| `Rain_log` | log1p of seasonal rainfall (Nov–Apr), ERA5-Land | log(mm) |
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
> not stored separately. In the revised analysis (`revision_code/`), the
> per-district trend is a **Theil–Sen** trend fitted on the training years only.

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
- **Scenario B (NDVI sensitivity case):** the *same* SSP2-4.5 climate with NDVI regressed on projected LST.

> **Both scenarios share identical climate forcing**, including `LST_mean_C`; they differ only in
> `NDVI_mean` (and the derived `NDVI×VPD` interaction). The difference between them therefore
> measures sensitivity to the NDVI assumption, and is not an uncertainty bound.
>
> **Projected LST note.** CMIP6 provides near-surface *air* temperature, while the historical
> predictor is MODIS MOD11A2 *land-surface* temperature. Projected `LST_mean_C` is therefore
> constructed on the MODIS scale as the 2001–2020 district LST climatology plus the SSP2-4.5
> temperature anomaly for each projection year (division-mean ΔT = +0.48 °C over 2026–2030).

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

`Forecast_FrozenClimate_2001_2020.csv` — same columns as the forecast tables above, from a
counterfactual run in which the five climate predictors (`LST_mean_C`, `Rain_log`, `VPD_kPa`,
`HeatStress_log`, `ColdStress_log`) are held at their 2001–2020 district means while `NDVI_mean`
and the frozen 2025 trend are unchanged. The difference from the Scenario A run gives the
attributable climate effect.

`Forecast_Uncertainty_Table.csv` — per-district summary of the Scenario A vs B spread.

| Column | Description | Unit |
|---|---|---|
| `District` | District name | — |
| `ScenA_mean`, `ScenB_mean` | 2026–2030 mean projected yield | kg/ha |
| `ScenA_min`, `ScenA_max` | Range of Scenario A across 2026–2030 | kg/ha |
| `A_minus_B` | Scenario A minus Scenario B | kg/ha |
| `A_minus_B_pct` | Same difference relative to Scenario B | percentage points |

`Forecast_SERWI_Weights.csv` — the per-district SERWI weights and bias corrections applied
when constructing the hybrid forecasts.

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

`DM_Test_Results_HLN_BH.csv` — the same 21 comparisons with small-sample and multiplicity
corrections applied.

| Column | Description |
|---|---|
| `DM_HLN` | DM statistic after the Harvey–Leybourne–Newbold small-sample correction (n = 20, h = 1) |
| `p_HLN` | Two-sided p-value from a t distribution with n − 1 degrees of freedom |
| `p_BH` | Benjamini–Hochberg false-discovery-rate adjusted p-value across all 21 tests |
| `BH_significant` | Yes/No at an FDR of 0.05 |
| `Nested` | True where one model contains the other as a component, so the two forecasts are not independent |
| `Conclusion` | Plain-language reading of the corrected result |

---

## 6. Intermediate arrays and model objects
- `*_y_pred_*`, `*_y_test`, `*_y_train`, `*_yr_test`, `*_dist_test`, `*_trend_*` (`.npy`)
  — per-model predictions, targets, year/district indices, and detrending components
  saved by the standalone notebooks and consumed by the SERWI hybrid notebooks.
- `xgb_wheat_model.pkl`, `lstm_wheat_model.keras`, `tcn_wheat_model.keras` — trained models.
- `*_x_scaler.pkl`, `*_y_scaler.pkl`, `xgb_feature_cols.pkl` — feature/target scalers and column order.
- `*_trend_models.pkl` — per-district linear trend models (for detrending / re-adding trend).

---

## 7. Revision files (`revision_code/code/`)
Scripts that reproduce every number in the revised manuscript. Run order is given in
`README.md`, `README_round2.md` and `README_round3.md` in that folder.
Protocol: train 2002–2018, validation 2019–2020, test 2021–2025 (used once);
per-district Theil–Sen trend; models selected on 2014–2020 rolling forecasts.

### 7.1 `NDVI_FixedMask.csv` — NDVI over a fixed wheat mask
100 rows = 4 districts × 25 years. Produced by
`wheat-yield-prediction/Gee_scripts/03_GEE_wheat_mask_NDVI_2001-2025.txt`.

| Column | Description | Unit |
|---|---|---|
| `Year` | Harvest year | year |
| `District` | District name | — |
| `NDVI_fixed` | Mean MODIS MOD13Q1 NDVI (250 m, SummaryQA ≤ 1), 1 Mar–30 Apr, over the fixed wheat mask | index (0–1) |
| `Fixed_area_ha` | Area of the fixed wheat mask within the district (same every year) | ha |

> **Fixed wheat mask.** A pixel is classed as wheat in a given year if NDVI at sowing
> (15 Oct–31 Dec, minimum) < 0.35, peak NDVI (15 Jan–20 Mar, maximum) ≥ 0.50,
> post-harvest NDVI (1 May–15 Jun, minimum) < 0.35, and peak − sowing ≥ 0.25.
> The fixed mask keeps pixels classed as wheat in at least 10 of the 20 pre-test
> years (2001–2020), so no information from the 2021–2025 test period is used.

### 7.2 `NDVI_WheatMask.csv` — NDVI over a yearly wheat mask
Same script and rule as 7.1, but the mask is rebuilt each year.

| Column | Description | Unit |
|---|---|---|
| `Year`, `District` | As above | — |
| `NDVI_wheat` | Mean Mar–Apr NDVI over that year's wheat mask | index (0–1) |
| `Wheat_area_ha` | Area classed as wheat in that year | ha |

### 7.3 `Tmax.csv` — mean daily maximum temperature

| Column | Description | Unit |
|---|---|---|
| `Year`, `District` | As above | — |
| `Tmax_mean_C` | Mean daily maximum 2 m air temperature, 1 Mar–30 Apr, ERA5-Land daily aggregates | °C |

### 7.4 `roll_baselines.csv` — one-year-ahead rolling forecasts, 2014–2025
48 rows = 4 districts × 12 years. Produced by `roll_baselines.py` (tune on years up to
Y−3, refit on years up to Y−1, forecast year Y). All yield columns in **kg/ha**.

| Column | Description |
|---|---|
| `District`, `Year` | District and forecast year |
| `obs` | Observed yield |
| `XGB` | XGBoost forecast (Theil–Sen trend) |
| `Trend only (Theil–Sen)`, `Trend only (OLS)`, `Trend only (recent window)` | Trend-only forecasts |
| `Ridge`, `Elastic net`, `Random forest`, `SVR (RBF)` | Other learners on the same features |
| `AR(1) residual`, `Persistence`, `Climatology` | Simple time-series baselines |
| `XGB (OLS trend)` | XGBoost with an OLS trend instead of Theil–Sen |

### 7.5 `roll_baselines_table.csv` — RMSE summary
One row per model; columns give RMSE (kg/ha) for `pre-test 2014-2020`,
`test 2021-2025 (rolling)` and `all 2014-2025`.
