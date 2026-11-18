# Data Cleaning & EDA for Data Analysis

Cleaning a messy **house-sales dataset** (1,200 rows, 36 columns), exploring it visually, engineering features, and checking whether the cleaned data is actually useful for modelling.

## What's inside

| File | Description |
|---|---|
| `clean_O.ipynb` | Full working notebook: cleaning, EDA, outlier checks, feature engineering, validation |
| `clean_O_resolved.ipynb` | **Final, tidy version** – sectioned, leakage-free, with modelling baseline and evaluation |
| `train_o.csv` | Raw messy house data (1,200 rows, 36 columns) |
| `clean_train.csv` | Cleaned output with engineered features (1,171 rows, 40 columns) |

> Start with **`clean_O_resolved.ipynb`**.

## Workflow (`clean_O_resolved.ipynb`)

1. **Load & inspect** – `head`, `info`, `dtypes`, parse `SaleDate` to datetime.
2. **Null checks & cleaning**
   - Standardise text (`HeatingQC` strip + title case), cast `KitchenQual` to `string`.
   - Treat `0` in `LotFrontage` as missing.
   - Deterministic fills: `MasVnrArea`, `TotalBsmtSF`, `PoolArea` → `0`; `PoolQC` → "No Pool"; `Alley` → "No Alley"; `Fence` → "No Fence"; `GarageType` → "No Garage"; `GarageYrBlt` → `YearBuilt`.
   - Statistical imputation (median/mode) is deferred to the modelling pipeline to **avoid data leakage**.
3. **EDA**
   - Missing-value heatmap and missing-% table
   - Histograms/KDE for numeric columns, count plots for categoricals
   - Scatter plots: `GrLivArea`, `LotArea`, `GarageArea` vs `SalePrice`
   - Correlation heatmap, box plots (`HouseStyle`, `OverallQual`, `Neighborhood` vs `SalePrice`)
   - IQR-based outlier counts
4. **Feature engineering**
   - `HouseAge = YrSold - YearBuilt`
   - `RemodelAge = YrSold - YearRemodAdd`
   - `TotalBathrooms = FullBath + 0.5 * HalfBath`
   - `TotalArea = TotalBsmtSF + GrLivArea`
   - Rows with impossible dates (sold before built/remodelled) are inspected and dropped rather than silently clipped.
5. **Validation** – `assert` checks and a `validate(df)` function (no duplicates, positive prices/areas, integer year columns, no NaNs in engineered features).
6. **Export & modelling baseline** – cleaned data saved to `clean_train.csv`; sklearn pipelines predict `log1p(SalePrice)` using 5-fold CV and a single holdout evaluation, followed by permutation importance and residual analysis.

## Results

Models: Dummy (mean), RidgeCV, RandomForest, HistGradientBoosting.

**5-fold CV (train split)**

| Model | R² (log) | RMSE (log) | MAE ($) |
|---|---|---|---|
| HistGradBoosting | 0.235 | 0.284 | 63,586 |
| Ridge | 0.225 | 0.286 | 64,049 |
| RandomForest | 0.221 | 0.287 | 64,412 |
| Dummy (mean) | -0.009 | 0.327 | 72,799 |

**Holdout (20%)**

| Model | R² (log) | MAE ($) |
|---|---|---|
| HistGradBoosting | 0.224 | 63,112 |
| Dummy (mean) | -0.008 | 73,911 |

The best model beats the baseline, but the score is low overall. EDA shows weak correlation between most features and price (e.g. `GrLivArea` ≈ 0.25, `LotArea` ≈ 0), which points to noisy, likely synthetic data. Model choice matters less than better features (e.g. ordinal-encoded quality columns, pool/garage/fireplace flags).

## Tech stack

Python 3 · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter

## Getting started

```bash
git clone https://github.com/Nirmalya625/data_cleaning-and-eda-for-data-analysis.git
cd data_cleaning-and-eda-for-data-analysis
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook clean_O_resolved.ipynb
```

The notebooks read the raw file as `train.o.csv`; the file in this repo is named `train_o.csv`. Update the `pd.read_csv(...)` path (or rename the file) before running.

## Key takeaways

- Decide what a missing value *means* (no pool vs. unknown) before choosing a fill.
- Fit imputers inside the pipeline on training data only.
- Inspect impossible values instead of clipping them blindly.
- Always compare against a dummy baseline.

## Author

**Nirmalya** – [GitHub](https://github.com/Nirmalya625)
