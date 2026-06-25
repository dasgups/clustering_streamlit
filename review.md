# Code Review: Streamlit Rebate Optimization App

## Summary

This repository has a strong implementation that generally aligns with the task separation: `build_pca_output.py` handles preprocessing/PCA and `main.py` handles Streamlit UI, clustering, and rebate analysis.

The review below highlights areas where the implementation is correct, plus opportunities to improve correctness, data handling, and robustness.

## What is working well

- `main.py` cleanly separates pricing and data science dashboards.
- PCA-based clustering is only run from saved `PC1`-`PC5` values in the app, not recomputed in `main.py`.
- The pricing workflow supports optional original and PCA file uploads and validates row count consistency.
- KMeans uses `n_clusters` of 3, 4, or 5, matching the requirement.
- Rebate gap and financial opportunity are computed in the app with a z-score outlier flag.
- The Data Science dashboard supports multiple models and preserves historical model scores until PCA input changes.
- `build_pca_output.py` uses log-transform and one-hot encoding before scaling and PCA, which follows the notebook methodology.
- PCA output includes enough components to reach at least 90% cumulative explained variance, with a minimum of five components.

## Key issues and recommended changes

### 1. `build_target_customer_table` uses the same column for ID and name

- In `main.py`, `build_target_customer_table()` constructs both `Customer ID` and `Customer Name` from the same `CUSTOMER_COL` value.
- This will not provide distinct identifier and name fields if the dataset actually has separate customer ID and customer name columns.

Recommendation:
- Use separate dataset columns for customer ID and customer name if available.
- If the dataset only exposes one customer field, adjust labels and report expectations accordingly.

### 2. Data science metadata is not loaded from uploaded PCA files

- `render_data_science_dashboard()` loads `PCA_METADATA_PATH` from disk even when the user uploads a custom PCA CSV.
- If a user uploads an alternate PCA output, the app may show stale or mismatched PCA metadata.

Recommendation:
- Load metadata from the uploaded PCA source when available, or at least warn that metadata is only sourced from the default `pca_metadata.csv`.
- Better: infer explained variance from the uploaded PCA file or require a matching metadata upload.

### 3. `CUSTOMER_COL` is hard-coded to `parent_name`

- The app assumes the customer identifier column is `parent_name`.
- If the raw dataset uses different identifier fields such as `Customer ID`, `Customer Name`, or similar, validation may miss this mismatch.

Recommendation:
- Add a more flexible identifier detection strategy or document that `parent_name` is required.
- Consider supporting both a customer ID column and a customer name column.

### 4. PCA builder may not exclude all identifier columns

- `build_pca_output.py` excludes `parent_name` and several rebate-related columns, but not all potential customer identifier fields.
- Requirement language mentions excluding `Customer ID`, `Customer Name`, `Rebate`, `Financial Opportunity`, and `Cluster`.

Recommendation:
- Confirm the raw dataset’s identifier fields and explicitly exclude them from PCA feature preparation.
- Add any additional identifier columns used by the dataset.

### 5. `load_default_pca_metadata()` is cached with file attributes only

- The metadata cache depends on `file_size` and `modified_time`, which is fine, but the function signature is somewhat indirect.
- This is not a bug, but the code path could be clearer if metadata loading were grouped with source input handling.

Recommendation:
- Consider unifying PCA metadata loading logic with PCA file loading so the app has one coherent data source path.

### 6. `run_analysis()` and `evaluate_data_science_model()` use `prepare_pca_features()` on the same DataFrame

- The code properly coerces PCA values to numeric and fills missing values.
- However, if PCA input contains unexpected non-numeric values, the silent coercion to zero may mask data problems.

Recommendation:
- Consider logging or alerting when PCA columns contain non-numeric values or NaNs before silently replacing them.

## Minor improvements

- The Data Science model comparison currently defaults to `KMeans` and `Gaussian Mixture`, which is a good UX choice.
- The app currently supports cluster counts 2 through 10 in the Data Science dashboard, even though the pricing dashboard is limited to 3-5 as required. That is probably okay, but the pricing workflow should remain constrained.
- `describe_cluster_characteristics()` uses portfolio medians to summarize clusters; that is a good business-friendly approach.

## Suggested review actions

1. Verify the actual raw dataset column names for customer identifiers and fix `CUSTOMER_COL` usage in `main.py`.
2. Fix the target report so `Customer ID` and `Customer Name` are populated correctly and distinctly.
3. Make PCA metadata loading aware of uploaded PCA files or document the limitation clearly.
4. Confirm `build_pca_output.py` excludes all identifier columns that should not be part of clustering.
5. Add a data validation step for PCA columns before silent numeric coercion.

## Conclusion

The app is structurally sound and close to the intended design, but it has a few practical data-handling issues to resolve before it can be considered fully reliable.

Addressing the customer ID/name handling and the PCA metadata/upload consistency will be the highest-priority fixes.
