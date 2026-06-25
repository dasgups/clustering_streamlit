# AGENTS.md

## Current Architecture

This project uses a two-step workflow:

1. `build_pca_output.py` regenerates `pca_5d_output.csv` and `pca_metadata.csv` from `segmentation_input_updated.csv` when the original data changes. The PCA output should include at least five components and continue through the component that reaches at least 90% cumulative explained variance.
2. `main.py` is the Streamlit app. It routes Pricing Team users to the rebate opportunity dashboard and Data Science Team users to PCA/model diagnostics.

Do not rerun preprocessing, encoding, scaling, or PCA inside the Streamlit app. Those steps belong in `build_pca_output.py` only.

Run the PCA builder on demand:

```powershell
python build_pca_output.py
```

Run the app:

```powershell
streamlit run main.py
```

The app should show a GIPP Team login screen before any dashboard data loads. Use `streamlit-authenticator` with hashed credentials from `st.secrets["auth"]`; do not store plaintext passwords in code.
Add a separate Data Science Team login user that reads the PCA output, shows PCA metadata, and compares clustering models.

## Project Overview

Build a Streamlit application for customer rebate optimization.

The app will help business users identify customers receiving higher rebates than comparable customers in the same cluster. These customers will be flagged as renewal targets so rebates can be revised and unnecessary spend can be reduced.

The app should stay lightweight during user interaction. It should only rerun KMeans and rebate calculations when the selected cluster count or input dataset changes.

Use Streamlit caching where applicable:

* Cache default CSV loading with `st.cache_data`.
* Cache uploaded CSV parsing with `st.cache_data`.
* Cache original/PCA dataframe combining with `st.cache_data`.
* Cache KMeans/rebate analysis results with `st.cache_data`, keyed by dataset contents and cluster count.
* Do not cache mutable fitted sklearn model objects with `st.cache_resource`; create fresh estimator instances before fitting.
* Do not rerun analysis for unchanged datasets and unchanged cluster counts.

Streamlit application code is in:

```text
main.py
```

Preprocessing and PCA generation code is in:

```text
build_pca_output.py
```

---

## Reference Files

### Research Notebook

Use the notebook below as the source of truth for methodology:

```text
Segmentation PCAdummy.ipynb
```

The notebook contains the research, data transformation steps, PCA logic, clustering approach, rebate gap calculation, outlier logic, and financial opportunity analysis.

### Mock Data

Use the raw mock data as the source dataset for PCA generation:

```text
segmentation_input_updated.csv
```

The Streamlit app should not load this raw file for modelling during normal use. Instead, `build_pca_output.py` should convert it into:

```text
pca_5d_output.csv
```

The Streamlit app should load `segmentation_input_updated.csv` and `pca_5d_output.csv`, unless the user uploads replacement files in the sidebar.

---

## Business Objective

The goal is to understand where customers are being given more rebate than expected.

The process is:

1. Cluster similar customers.
2. Calculate the median rebate for each cluster.
3. Compare each customer's rebate against their cluster median.
4. Identify outlier customers with higher-than-expected rebates.
5. Calculate the financial opportunity from correcting those rebates during renewal.

---

## Application Requirements

## File Structure

The project should be simple and contain:

```text
main.py
build_pca_output.py
Segmentation PCAdummy.ipynb
segmentation_input_updated.csv
pca_5d_output.csv
pca_metadata.csv
AGENTS.md
.gitignore
```

All Streamlit UI, KMeans clustering, rebate calculations, charts, and report generation should be implemented inside:

```text
main.py
```

All preprocessing, feature preparation, scaling, and PCA generation should be implemented inside:

```text
build_pca_output.py
```

Do not add more Python scripts unless explicitly requested later.

---

# Python File Requirements

## `main.py` Required Libraries

```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances, silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns
```

`main.py` also uses:

```python
import streamlit_authenticator as stauth
```

## `build_pca_output.py` Required Libraries

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

`main.py` should not import `StandardScaler` or `PCA`. Those imports belong in `build_pca_output.py`.

Optional for future UI work:

```python
import plotly.express as px
```

---

# Data Processing Logic

This logic belongs in `build_pca_output.py`, not `main.py`.

## 1. Load Raw Data

The PCA builder should load the raw CSV:

```python
df = pd.read_csv("segmentation_input_updated.csv")
```

The original dataframe must be preserved and saved alongside the PCA output for reporting.

`main.py` should load the original data and the prepared PCA output:

```python
original_df = pd.read_csv("segmentation_input_updated.csv")
pca_df = pd.read_csv("pca_5d_output.csv")
```

`main.py` may include optional uploaders for both files. The original upload should be the unscaled business file. The PCA upload should be a prepared PCA output CSV containing `PC*` component columns. The Pricing Team dashboard requires at least `PC1` through `PC5`; the Data Science dashboard should detect all available `PC*` columns. The app should combine original and PCA files by row order and stop with an error if row counts differ.

---

## 2. Feature Preparation

Use the methodology from:

```text
Segmentation PCAdummy.ipynb
```

The PCA builder should follow the notebook logic as closely as possible.

Steps:

1. Identify clustering features.
2. Apply log transformation to skewed numeric columns.
3. Encode categorical variables.
4. Scale the transformed dataset.
5. Run PCA.
6. Save at least the first five principal components and continue through the component that reaches at least 90% cumulative explained variance in `pca_5d_output.csv`.

During Pricing Team Streamlit interaction, use the saved `PC1` through `PC5` columns directly for clustering. During Data Science Team model comparison, allow the user to choose how many available PCA components to include.

---

## 3. Log Transformation

Because the data is highly skewed, apply log transformation to relevant numeric columns.

Use:

```python
np.sign(x) * np.log1p(np.abs(x))
```

This avoids issues with zero values and handles any negative values safely.

---

## 4. Encoding

Encode categorical fields before scaling and PCA.

Recommended approach:

```python
pd.get_dummies()
```

The encoded dataset should only include fields intended for clustering.

Do not include customer identifier fields directly in clustering.

Examples of fields to exclude from clustering:

```text
Customer ID
Customer Name
Rebate
Financial Opportunity
Cluster
```

Actual exclusions should follow the notebook.

---

## 5. Scaling

Apply standard scaling:

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)
```

---

## 6. PCA

Apply PCA and retain enough components to reach at least 90% cumulative explained variance, with a minimum of five components:

```python
full_pca = PCA(random_state=1)
full_pca.fit(scaled_data)
cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
n_components = max(5, np.searchsorted(cumulative_variance, 0.90) + 1)

pca = PCA(n_components=n_components, random_state=1)
pca_data = pca.fit_transform(scaled_data)
```

Create a PCA dataframe:

```python
component_columns = [f"PC{i}" for i in range(1, n_components + 1)]
pca_df = pd.DataFrame(
    pca_data,
    columns=component_columns
)
```

Also save `pca_metadata.csv` with component-level and cumulative explained variance.

---

## 7. KMeans Clustering

Run KMeans on:

```text
PC1, PC2, PC3, PC4, PC5
```

The user should be able to select the number of clusters from:

```text
3, 4, 5
```

Example:

```python
n_clusters = st.sidebar.selectbox(
    "Select Number of Clusters",
    [3, 4, 5]
)
```

Use:

```python
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)
```

Attach cluster labels back to the original dataframe.

Calculate the silhouette score on the same `PC1` through `PC5` matrix used for KMeans. Show it only as a small reference caption, not as a prominent KPI card.

For large datasets, calculate silhouette on a fixed random sample to keep interactions responsive.

---

# Rebate Analysis Logic

## Cluster Median Rebate

For each cluster, calculate:

```text
cluster_benchmark_cpl = median(overall_median_rebate_cpl) by Cluster
```

Then map it back to each customer.

---

## Rebate Gap

Calculate:

```text
rebate_gap_cpl = overall_median_rebate_cpl - cluster_benchmark_cpl
```

Customers with positive rebate gap are receiving more rebate than the cluster benchmark.

---

## Financial Opportunity

Calculate the potential savings opportunity using the notebook formula:

```text
rebate_opportunity_gbp = rebate_gap_cpl * total_vol / 100
```

Clip the result at zero so only positive recovery is counted.

---

## Outlier Detection

Identify customers who are high-rebate outliers within each cluster.

Use the same z-score outlier methodology from:

```text
Segmentation PCAdummy.ipynb
```

Calculate:

```text
z_score = rebate_gap_cpl / cluster_std
```

Flag customers where:

```text
z_score > 1.5
```

These customers should have `rebate_flag = "Over Rebated"` and are the target customers for rebate revision.

Optionally mark customers where `z_score < -1.5` as `Under Rebated`. All other customers should be `In Range`.

---

# Streamlit App Layout

## Login

Show the main login screen branded as:

```text
GIPP Team
```

After login, show the current team name. Use separate users for:

```text
Pricing Team
Data Science Team
```

The dashboard should not load or display until the user signs in.

Use `streamlit-authenticator` for login and logout. Do not store plaintext passwords in `main.py`.

Credential source:

```toml
[auth]
cookie_name = "rebate_optimizer_auth"
cookie_key = "long_random_cookie_key"
cookie_expiry_days = 1

[auth.credentials.usernames.pricing_team]
email = "pricing.team@example.com"
name = "Pricing Team"
password = "bcrypt_hash_only"
```

Generate password hashes with:

```powershell
python -c "import streamlit_authenticator as stauth; print(stauth.Hasher.hash('your_password'))"
```

Include a sidebar sign-out control after login.

Pricing Team users should see the rebate optimization dashboard.

Data Science Team users should see a diagnostics dashboard that:

* Loads `pca_5d_output.csv`.
* Shows number of PCA components.
* Shows explained variance from `pca_metadata.csv`.
* Allows an optional matching PCA metadata upload when a replacement PCA output is uploaded.
* Allows the user to select cluster counts.
* Allows the user to select how many PCA components to include in model comparison.
* Runs selected model comparisons for KMeans, Gaussian Mixture, local KMedoids, and Agglomerative Clustering.
* Shows a results table with cluster size, model name, cluster count, PCA component count, and silhouette score.
* Preserves past model score rows while the PCA output file is unchanged.
* Clears stored model scores automatically when the PCA output changes, and provide a manual clear button.

## Sidebar

The sidebar should include:

1. Cluster count selector.
2. Run analysis button.
3. Optional original CSV uploader.
4. Optional PCA output CSV uploader.
5. Optional filters for cluster or customer segment.

The original uploader should accept the unscaled business file. If no original file is uploaded, use `segmentation_input_updated.csv`.

The PCA uploader should accept prepared PCA output files containing `PC*` component columns. Pricing requires at least `PC1` through `PC5`. If no PCA file is uploaded, use `pca_5d_output.csv`.

In the Data Science dashboard, a replacement PCA upload can be paired with an optional replacement metadata CSV. If a replacement PCA file is uploaded without metadata, do not show default explained-variance metadata as if it belonged to the uploaded file.

Example:

```python
st.sidebar.title("Controls")

n_clusters = st.sidebar.selectbox(
    "Select Number of Clusters",
    [3, 4, 5]
)

run_analysis = st.sidebar.button("Run Analysis")
```

---

# Main Page

App title:

```text
Customer Rebate Optimization Dashboard
```

Intro text:

```text
This app identifies customers receiving higher rebates than comparable customers in the same cluster and estimates potential rebate savings during renewal.
```

Use two tabs:

```python
tab1, tab2 = st.tabs([
    "Cluster Analysis",
    "Rebate Opportunity Report"
])
```

---

# Tab 1: Cluster Analysis

## Purpose

This tab helps business users understand how customers have been grouped and what the cluster characteristics are.

## Required Outputs

### 1. Cluster Statistics Table

Show statistics against the original data after cluster assignment.

Use table styling to highlight the minimum and maximum values in each numeric metric column so cluster differences are easy to scan.

Include:

```text
Cluster
Customer Count
Median Rebate
Mean Rebate
Minimum Rebate
Maximum Rebate
Standard Deviation Rebate
Total Financial Opportunity
```

Also include other business metrics from the mock data where relevant.

---

### 2. Boxplots

Show boxplots by cluster to explain cluster characteristics.

Use a three-column chart grid. Include rebate and important numeric business variables in the same grid rather than showing rebate separately.

At minimum include:

```text
Rebate by Cluster
```

Also include boxplots for important numeric variables used in clustering.

Examples:

```text
Revenue by Cluster
Margin by Cluster
Volume by Cluster
Spend by Cluster
```

Actual variable names should match the saved PCA output CSV.

---

### 3. Cluster Characteristics

Add an expander under the Cluster Analysis tab that briefly describes every cluster using original unscaled business columns.

Descriptions should be concise and business-readable. Include:

```text
Customer count
Top 2-3 metrics that are meaningfully higher or lower than the portfolio median
Number of over-rebated customers
Total financial opportunity
```

Do not include the PCA visualization unless explicitly requested later.

---

# Tab 2: Rebate Opportunity Report

## Purpose

This tab identifies customers that should be reviewed during renewal.

These are customers whose rebates are high compared to similar customers in the same cluster.

## Required Outputs

### 1. Opportunity Summary

Show:

```text
Cluster
Number of Outlier Customers
Total Financial Opportunity
Average Rebate Gap
Median Cluster Rebate
```

---

### 2. Target Customer Table

Show all outlier customers across every cluster.

Required columns:

```text
Customer Name
Cluster
Current Rebate
Cluster Median Rebate
Rebate Gap
Financial Opportunity
Outlier Flag
```

Use the actual column names from the CSV where different. For the current mock data, customer name is unique and is the customer key for the report. If a real customer ID field is added later, route that field to a `Customer_ID` report column and keep `Customer Name` as the readable customer label.

Sort by:

```text
Financial Opportunity descending
```

---

### 3. Cluster Filter

Allow the user to filter the opportunity report by cluster.

Example:

```python
selected_cluster = st.selectbox(
    "Select Cluster",
    ["All"] + sorted(df["Cluster"].unique().tolist())
)
```

---

### 4. Download Report

Provide a CSV download of the target customer table.

Example:

```python
st.download_button(
    label="Download Rebate Opportunity Report",
    data=report_csv,
    file_name="rebate_opportunity_report.csv",
    mime="text/csv"
)
```

---

# Expected User Flow

1. If `segmentation_input_updated.csv` has changed, run `python build_pca_output.py`.
2. The builder creates or refreshes `pca_5d_output.csv`.
3. User opens the Streamlit app with `streamlit run main.py`.
4. App loads `segmentation_input_updated.csv` and `pca_5d_output.csv`, unless the user uploads replacement files.
5. User selects number of clusters: 3, 4, or 5.
6. App runs KMeans on `PC1` through `PC5`.
7. App displays cluster statistics, cluster characteristics, and boxplots from the original unscaled business columns in Tab 1.
8. App displays rebate gap, outlier customers, and financial opportunity in Tab 2.
9. User downloads the opportunity report.
10. Business team uses the report for renewal rebate correction.

---

# Implementation Notes

## Important Rules

* Use `Segmentation PCAdummy.ipynb` as the methodology reference.
* Use `segmentation_input_updated.csv` as the raw input to `build_pca_output.py`.
* Use `segmentation_input_updated.csv` and `pca_5d_output.csv` as the default app inputs to `main.py`.
* If files are uploaded in the app, the original file must be unscaled business data and the PCA file must already contain enough `PC*` component columns for the selected workflow.
* Uploaded original and PCA files must have the same number of rows and be in the same row order.
* Put Streamlit UI, KMeans clustering, rebate analysis, charts, and downloads inside `main.py`.
* Put preprocessing, encoding, scaling, and PCA generation inside `build_pca_output.py`.
* Do not rerun preprocessing, encoding, scaling, or PCA inside `main.py`.
* Preserve the original dataframe for reporting.
* Do not cluster on customer ID, customer name, parent name, or other identifier columns.
* For now, `parent_name` or `customer_name` should populate `Customer Name` in the target report because customer name is unique. If a real ID field is added later, use it for a separate `Customer_ID` column.
* PCA component columns must be numeric and non-missing. Do not silently convert malformed PCA values to zero inside `main.py`.
* Use PCA components `PC1` through `PC5` for KMeans.
* Allow only 3, 4, or 5 clusters.
* After cluster labels are generated, calculate cluster analysis, rebate statistics, and boxplots against original unscaled business columns, not transformed or PCA values.
* Do not show PCA values directly in Tab 1 unless explicitly requested later.
* Use cluster median rebate as the benchmark.
* Outlier customers are the target customers for renewal rebate revision.
* Keep `.streamlit/secrets.toml`, `__pycache__/`, local Streamlit logs, and `.venv/` out of git. They are ignored in `.gitignore`.
* If authentication settings need to be shared, create a redacted example file rather than committing real secrets or password hashes.

## Current Code Organization

`main.py` is organized around small helper functions:

* Authentication helpers: `to_plain_dict()`, `get_authenticator()`, and `require_login()`.
* Data loading and validation helpers: `load_csv()`, `combine_original_and_pca_data()`, `validate_input()`, and `prepare_pca_features()`.
* Pricing workflow helpers: `render_pricing_dashboard()`, `render_pricing_controls()`, `get_or_run_pricing_analysis()`, `render_cluster_analysis_tab()`, and `render_opportunity_report_tab()`.
* Data Science workflow helpers: `render_data_science_dashboard()`, `run_data_science_models()`, and `evaluate_data_science_model()`.

Keep future changes inside these helpers where practical instead of adding new top-level procedural blocks at the bottom of `main.py`.

---

# Success Criteria

The final Streamlit app should:

1. Run from `main.py`.
2. Load `segmentation_input_updated.csv` and `pca_5d_output.csv` by default, with optional upload paths for replacement original and PCA output CSV files.
3. Avoid rerunning preprocessing and PCA during Streamlit interactions.
4. Allow the user to choose 3, 4, or 5 clusters.
5. Generate customer clusters using KMeans on `PC1` through `PC5`.
6. Show silhouette score as a small reference.
7. Show cluster statistics in Tab 1.
8. Show a cluster characteristics expander in Tab 1.
9. Show boxplots in Tab 1.
10. Show rebate gap analysis in Tab 2.
11. Identify outlier customers in every cluster.
12. Show financial opportunity.
13. Allow download of the target customer report.
14. Route Data Science Team users to PCA/model diagnostics.
15. Show Data Science model comparison results for KMeans, Gaussian Mixture, local KMedoids, and Agglomerative Clustering when selected.

The final preprocessing workflow should:

1. Run from `build_pca_output.py`.
2. Load `segmentation_input_updated.csv`.
3. Use the notebook methodology.
4. Preserve original business columns.
5. Save `pca_5d_output.csv` with enough `PC*` columns to reach at least 90% cumulative explained variance, with a minimum of five components.
6. Save `pca_metadata.csv` with explained variance by component.
