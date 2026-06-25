import io
import math
import os
from collections.abc import Mapping

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances, silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns


ORIGINAL_DATA_PATH = "segmentation_input_updated.csv"
PCA_DATA_PATH = "pca_5d_output.csv"
PCA_METADATA_PATH = "pca_metadata.csv"
REBATE_COL = "overall_median_rebate_cpl"
VOLUME_COL = "total_vol"
CUSTOMER_NAME_COLUMNS = [
    "customer_name",
    "Customer Name",
    "parent_name",
    "Parent Name",
]
CUSTOMER_ID_COLUMNS = [
    "Customer_ID",
    "customer_id",
    "Customer ID",
]
PCA_COLUMNS = ["PC1", "PC2", "PC3", "PC4", "PC5"]
SILHOUETTE_SAMPLE_SIZE = 10_000
KMEDOIDS_SAMPLE_SIZE = 20_000
AGGLOMERATIVE_SAMPLE_SIZE = 15_000

BUSINESS_METRIC_COLUMNS = [
    "total_vol",
    "total_dpm",
    "national_volume",
    "international_volume",
    "supplied_by_bp_vol",
    "non_supplied_by_bp_vol",
    "used_retail_sites_count",
    "total_rebate_amount",
]

DISPLAY_LABELS = {
    REBATE_COL: "Median Rebate cpl",
    "total_vol": "Total Volume",
    "total_dpm": "Total DPM",
    "national_volume": "National Volume",
    "international_volume": "International Volume",
    "used_retail_sites_count": "Retail Sites",
    "total_rebate_amount": "Total Rebate GBP",
}


st.set_page_config(
    page_title="Customer Rebate Optimization Dashboard",
    layout="wide",
)

sns.set_theme(style="whitegrid")


def to_plain_dict(value):
    if isinstance(value, Mapping):
        return {key: to_plain_dict(nested_value) for key, nested_value in value.items()}
    return value


def get_authenticator() -> stauth.Authenticate:
    try:
        auth_config = to_plain_dict(st.secrets["auth"])
    except Exception:
        st.error(
            "Authentication is not configured. Add hashed credentials to "
            "`.streamlit/secrets.toml` under `[auth]`."
        )
        st.stop()

    credentials = auth_config.get("credentials")
    if not credentials:
        st.error("Missing `[auth.credentials]` in `.streamlit/secrets.toml`.")
        st.stop()

    return stauth.Authenticate(
        credentials,
        cookie_name=auth_config.get("cookie_name", "rebate_optimizer_auth"),
        cookie_key=auth_config.get("cookie_key"),
        cookie_expiry_days=float(auth_config.get("cookie_expiry_days", 1)),
        auto_hash=False,
    )


def require_login() -> tuple[stauth.Authenticate, str, str]:
    authenticator = get_authenticator()

    try:
        authenticator.login(location="unrendered", key="silent_login")
    except Exception:
        pass

    if st.session_state.get("authentication_status"):
        return (
            authenticator,
            st.session_state.get("name", "Pricing Team"),
            st.session_state.get("username", ""),
        )

    st.title("GIPP Team")
    st.subheader("Customer Rebate Optimization Dashboard")

    authenticator.login(
        location="main",
        fields={
            "Form name": "GIPP Team Login",
            "Username": "Username",
            "Password": "Password",
            "Login": "Sign In",
        },
    )

    authentication_status = st.session_state.get("authentication_status")
    if authentication_status:
        st.rerun()
    if authentication_status is False:
        st.error("Invalid username or password.")
    else:
        st.caption("Sign in to view cluster analysis and rebate opportunity reporting.")

    st.stop()


@st.cache_data(show_spinner=False)
def load_default_original_data(path: str, file_size: int, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_default_pca_data(path: str, file_size: int, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_default_pca_metadata(path: str, file_size: int, modified_time: float) -> pd.DataFrame:
    return pd.read_csv(path)


def file_cache_key(path: str) -> tuple[str, int, float]:
    stats = os.stat(path)
    return path, stats.st_size, stats.st_mtime


def load_csv(uploaded_file, default_loader, default_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return load_uploaded_csv(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    return default_loader(*file_cache_key(default_path))


@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def combine_original_and_pca_data(
    original_df: pd.DataFrame,
    pca_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(original_df) != len(pca_df):
        raise ValueError(
            "The original data and PCA output must have the same number of rows. "
            f"Original rows: {len(original_df):,}; PCA rows: {len(pca_df):,}."
        )

    missing_pca_columns = [column for column in PCA_COLUMNS if column not in pca_df.columns]
    if missing_pca_columns:
        raise ValueError(
            "The PCA output is missing required columns: "
            + ", ".join(missing_pca_columns)
        )

    original_without_pca = original_df.drop(columns=PCA_COLUMNS, errors="ignore")
    pca_columns = pca_df[PCA_COLUMNS].reset_index(drop=True)

    return pd.concat(
        [original_without_pca.reset_index(drop=True), pca_columns],
        axis=1,
    )


def validate_input(df: pd.DataFrame) -> list[str]:
    required_columns = [REBATE_COL, VOLUME_COL] + PCA_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if not get_customer_name_column(df):
        missing_columns.append("customer_name or parent_name")
    return missing_columns


def get_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def get_customer_name_column(df: pd.DataFrame) -> str | None:
    return get_first_existing_column(df, CUSTOMER_NAME_COLUMNS)


def get_customer_id_column(df: pd.DataFrame) -> str | None:
    return get_first_existing_column(df, CUSTOMER_ID_COLUMNS)


def get_available_pca_columns(df: pd.DataFrame) -> list[str]:
    pca_columns = [
        column
        for column in df.columns
        if column.startswith("PC") and column[2:].isdigit()
    ]
    return sorted(pca_columns, key=lambda column: int(column[2:]))


@st.cache_data(show_spinner=False)
def dataframe_signature(df: pd.DataFrame) -> tuple:
    hashed_rows = pd.util.hash_pandas_object(df, index=True)
    return (
        df.shape,
        tuple(df.columns),
        int(hashed_rows.sum()),
    )


def prepare_pca_features(
    df: pd.DataFrame,
    selected_pca_columns: tuple[str, ...] = tuple(PCA_COLUMNS),
) -> pd.DataFrame:
    missing_pca_columns = [
        column for column in selected_pca_columns if column not in df.columns
    ]
    if missing_pca_columns:
        raise ValueError(
            "The PCA output is missing required columns: "
            + ", ".join(missing_pca_columns)
        )

    pca_features = df[list(selected_pca_columns)].copy()
    invalid_columns = []
    for column in selected_pca_columns:
        numeric_values = pd.to_numeric(pca_features[column], errors="coerce")
        if numeric_values.replace([np.inf, -np.inf], np.nan).isna().any():
            invalid_columns.append(column)
        pca_features[column] = numeric_values

    if invalid_columns:
        raise ValueError(
            "PCA component columns must contain only numeric, non-missing values. "
            "Check: " + ", ".join(invalid_columns)
        )

    return pca_features


def make_kmeans(n_clusters: int) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )


def make_gaussian_mixture(n_clusters: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=n_clusters,
        random_state=42,
    )


@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, float]:
    reporting_df = df.copy()
    pca_features = prepare_pca_features(reporting_df)

    kmeans = make_kmeans(n_clusters)
    labels = kmeans.fit_predict(pca_features)
    score = silhouette_for_labels(pca_features, labels)

    reporting_df["Cluster"] = labels
    reporting_df = add_rebate_analysis(reporting_df)
    return reporting_df, float(score)


def format_cluster_sizes(labels: np.ndarray) -> str:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return ", ".join(
        f"{int(label)}: {int(count):,}"
        for label, count in zip(unique_labels, counts)
    )


def sample_features(
    pca_features: pd.DataFrame,
    sample_size: int,
    model_label: str,
) -> tuple[pd.DataFrame, str]:
    if len(pca_features) <= sample_size:
        return pca_features, f"Full dataset | {model_label}"

    sampled_features = pca_features.sample(
        n=sample_size,
        random_state=42,
    )
    return sampled_features, f"Sampled {sample_size:,} rows | {model_label}"


def initialize_kmedoids(data: np.ndarray, n_clusters: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    first_medoid = int(rng.integers(len(data)))
    medoids = [first_medoid]

    for _ in range(1, n_clusters):
        distances = pairwise_distances(data, data[medoids]).min(axis=1)
        distances[medoids] = -1
        medoids.append(int(np.argmax(distances)))

    return np.array(medoids, dtype=int)


def run_local_kmedoids(
    pca_features: pd.DataFrame,
    n_clusters: int,
    max_iter: int = 20,
) -> np.ndarray:
    data = pca_features.to_numpy(dtype=float)
    if len(data) < n_clusters:
        raise ValueError("KMedoids requires at least as many rows as clusters.")

    medoid_indices = initialize_kmedoids(data, n_clusters)
    labels = np.zeros(len(data), dtype=int)

    for _ in range(max_iter):
        medoid_distances = pairwise_distances(data, data[medoid_indices])
        labels = medoid_distances.argmin(axis=1)
        updated_medoids = medoid_indices.copy()

        for cluster_id in range(n_clusters):
            cluster_indices = np.flatnonzero(labels == cluster_id)
            if len(cluster_indices) == 0:
                updated_medoids[cluster_id] = int(
                    np.argmax(medoid_distances.min(axis=1))
                )
                continue

            cluster_data = data[cluster_indices]
            within_cluster_distances = pairwise_distances(cluster_data)
            best_cluster_position = int(within_cluster_distances.sum(axis=1).argmin())
            updated_medoids[cluster_id] = int(cluster_indices[best_cluster_position])

        if np.array_equal(updated_medoids, medoid_indices):
            break

        medoid_indices = updated_medoids

    return pairwise_distances(data, data[medoid_indices]).argmin(axis=1)


def silhouette_for_labels(features: pd.DataFrame, labels: np.ndarray) -> float:
    return float(
        silhouette_score(
            features,
            labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(features)),
            random_state=42,
        )
    )


def model_result_row(
    model_name: str,
    n_clusters: int,
    selected_pca_columns: tuple[str, ...],
    features: pd.DataFrame,
    labels: np.ndarray,
    notes: str,
) -> dict:
    return {
        "Model Name": model_name,
        "Clusters": n_clusters,
        "PCA Components": len(selected_pca_columns),
        "Cluster Size": format_cluster_sizes(labels),
        "Silhouette Score": round(silhouette_for_labels(features, labels), 3),
        "Notes": notes,
    }


def evaluate_data_science_model(
    model_name: str,
    pca_features: pd.DataFrame,
    n_clusters: int,
    selected_pca_columns: tuple[str, ...],
    kmedoids_sample_size: int,
    agglomerative_sample_size: int,
) -> dict:
    if model_name == "KMeans":
        labels = make_kmeans(n_clusters).fit_predict(pca_features)
        return model_result_row(
            model_name,
            n_clusters,
            selected_pca_columns,
            pca_features,
            labels,
            "Full dataset",
        )

    if model_name == "Gaussian Mixture":
        labels = make_gaussian_mixture(n_clusters).fit_predict(pca_features)
        return model_result_row(
            model_name,
            n_clusters,
            selected_pca_columns,
            pca_features,
            labels,
            "Full dataset",
        )

    if model_name == "Agglomerative Clustering":
        sampled_features, notes = sample_features(
            pca_features,
            agglomerative_sample_size,
            "Agglomerative",
        )
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(
            sampled_features
        )
        return model_result_row(
            model_name,
            n_clusters,
            selected_pca_columns,
            sampled_features,
            labels,
            notes,
        )

    if model_name == "KMedoids":
        sampled_features, notes = sample_features(
            pca_features,
            kmedoids_sample_size,
            "Local KMedoids",
        )
        labels = run_local_kmedoids(sampled_features, n_clusters)
        return model_result_row(
            model_name,
            n_clusters,
            selected_pca_columns,
            sampled_features,
            labels,
            notes,
        )

    raise ValueError(f"Unsupported model: {model_name}")


@st.cache_data(show_spinner=False)
def run_data_science_models(
    pca_df: pd.DataFrame,
    cluster_counts: tuple[int, ...],
    model_names: tuple[str, ...],
    selected_pca_columns: tuple[str, ...],
    kmedoids_sample_size: int,
    agglomerative_sample_size: int,
) -> pd.DataFrame:
    pca_features = prepare_pca_features(pca_df, selected_pca_columns)
    results = [
        evaluate_data_science_model(
            model_name,
            pca_features,
            n_clusters,
            selected_pca_columns,
            kmedoids_sample_size,
            agglomerative_sample_size,
        )
        for n_clusters in cluster_counts
        for model_name in model_names
    ]

    return pd.DataFrame(results)


def append_model_scores(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        return new_df

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    return (
        combined_df.drop_duplicates(
            subset=["Model Name", "Clusters", "PCA Components"],
            keep="last",
        )
        .sort_values(
            ["Silhouette Score", "PCA Components", "Clusters", "Model Name"],
            ascending=[False, True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def add_rebate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    analyzed_df = df.copy()
    analyzed_df[REBATE_COL] = pd.to_numeric(analyzed_df[REBATE_COL], errors="coerce")
    analyzed_df[VOLUME_COL] = pd.to_numeric(analyzed_df[VOLUME_COL], errors="coerce")

    analyzed_df["cluster_benchmark_cpl"] = analyzed_df.groupby("Cluster")[
        REBATE_COL
    ].transform("median")
    analyzed_df["cluster_std"] = analyzed_df.groupby("Cluster")[REBATE_COL].transform(
        "std"
    )
    analyzed_df["rebate_gap_cpl"] = (
        analyzed_df[REBATE_COL] - analyzed_df["cluster_benchmark_cpl"]
    )

    safe_std = analyzed_df["cluster_std"].replace(0, np.nan)
    analyzed_df["z_score"] = (analyzed_df["rebate_gap_cpl"] / safe_std).fillna(0)

    analyzed_df["rebate_flag"] = "In Range"
    analyzed_df.loc[analyzed_df["z_score"] > 1.5, "rebate_flag"] = "Over Rebated"
    analyzed_df.loc[analyzed_df["z_score"] < -1.5, "rebate_flag"] = "Under Rebated"

    analyzed_df["rebate_opportunity_gbp"] = (
        analyzed_df["rebate_gap_cpl"] * analyzed_df[VOLUME_COL] / 100
    ).clip(lower=0)

    safe_benchmark = analyzed_df["cluster_benchmark_cpl"].replace(0, np.nan)
    analyzed_df["rebate_variance_pct"] = (
        analyzed_df["rebate_gap_cpl"] / safe_benchmark * 100
    ).fillna(0)

    return analyzed_df


def build_cluster_statistics(df: pd.DataFrame) -> pd.DataFrame:
    customer_name_column = get_customer_name_column(df)
    grouped = df.groupby("Cluster")

    stats = grouped.agg(
        Customer_Count=(customer_name_column, "count"),
        Median_Rebate=(REBATE_COL, "median"),
        Mean_Rebate=(REBATE_COL, "mean"),
        Minimum_Rebate=(REBATE_COL, "min"),
        Maximum_Rebate=(REBATE_COL, "max"),
        Standard_Deviation_Rebate=(REBATE_COL, "std"),
        Total_Financial_Opportunity=("rebate_opportunity_gbp", "sum"),
    )

    available_metrics = [
        column
        for column in BUSINESS_METRIC_COLUMNS
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    if available_metrics:
        metric_summary = grouped[available_metrics].median().add_prefix("Median_")
        stats = stats.join(metric_summary)

    return stats.reset_index().round(2)


def highlight_min_max(data: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    numeric_columns = data.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        if column == "Cluster":
            continue

        min_value = data[column].min()
        max_value = data[column].max()
        styles.loc[data[column] == min_value, column] = (
            "background-color: #fde2e2; color: #7f1d1d; font-weight: 600;"
        )
        styles.loc[data[column] == max_value, column] = (
            "background-color: #dcfce7; color: #14532d; font-weight: 600;"
        )

    return styles


def describe_cluster_characteristics(df: pd.DataFrame) -> dict[int, list[str]]:
    metric_labels = {
        REBATE_COL: "median rebate",
        "total_vol": "volume",
        "total_dpm": "DPM",
        "national_volume": "national volume",
        "international_volume": "international volume",
        "used_retail_sites_count": "retail site count",
        "total_rebate_amount": "total rebate amount",
    }
    candidate_metrics = [
        column
        for column in metric_labels
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    cluster_medians = df.groupby("Cluster")[candidate_metrics].median()
    overall_medians = df[candidate_metrics].median()
    descriptions = {}

    for cluster, row in cluster_medians.iterrows():
        cluster_df = df[df["Cluster"] == cluster]
        statements = [
            f"{len(cluster_df):,} customers are in this cluster.",
        ]

        differences = []
        for column in candidate_metrics:
            overall_value = overall_medians[column]
            cluster_value = row[column]
            if pd.isna(overall_value) or pd.isna(cluster_value) or overall_value == 0:
                continue

            pct_difference = (cluster_value - overall_value) / abs(overall_value) * 100
            differences.append((abs(pct_difference), pct_difference, column, cluster_value))

        for _, pct_difference, column, cluster_value in sorted(
            differences,
            reverse=True,
        )[:3]:
            direction = "higher" if pct_difference > 0 else "lower"
            statements.append(
                f"{metric_labels[column].capitalize()} is {abs(pct_difference):.0f}% "
                f"{direction} than the portfolio median ({cluster_value:,.2f})."
            )

        target_count = int((cluster_df["rebate_flag"] == "Over Rebated").sum())
        opportunity = cluster_df.loc[
            cluster_df["rebate_flag"] == "Over Rebated",
            "rebate_opportunity_gbp",
        ].sum()
        statements.append(
            f"{target_count:,} customers are over-rebated, with GBP {opportunity:,.0f} "
            "of potential recovery."
        )

        descriptions[int(cluster)] = statements

    return descriptions


def render_cluster_characteristics(df: pd.DataFrame) -> None:
    descriptions = describe_cluster_characteristics(df)
    with st.expander("Cluster Characteristics", expanded=False):
        for cluster in sorted(descriptions):
            st.markdown(f"**Cluster {cluster}**")
            for statement in descriptions[cluster]:
                st.write(f"- {statement}")


def build_opportunity_summary(df: pd.DataFrame) -> pd.DataFrame:
    targets = df[df["rebate_flag"] == "Over Rebated"].copy()
    customer_name_column = get_customer_name_column(df)
    if targets.empty:
        return pd.DataFrame(
            columns=[
                "Cluster",
                "Number of Outlier Customers",
                "Total Financial Opportunity",
                "Average Rebate Gap",
                "Median Cluster Rebate",
            ]
        )

    summary = targets.groupby("Cluster").agg(
        **{
            "Number of Outlier Customers": (customer_name_column, "count"),
            "Total Financial Opportunity": ("rebate_opportunity_gbp", "sum"),
            "Average Rebate Gap": ("rebate_gap_cpl", "mean"),
            "Median Cluster Rebate": ("cluster_benchmark_cpl", "median"),
        }
    )
    return summary.reset_index().round(2)


def build_target_customer_table(df: pd.DataFrame) -> pd.DataFrame:
    targets = df[df["rebate_flag"] == "Over Rebated"].copy()
    targets = targets.sort_values("rebate_opportunity_gbp", ascending=False)
    customer_name_column = get_customer_name_column(targets)
    customer_id_column = get_customer_id_column(targets)

    report_columns = {}
    if customer_id_column:
        report_columns["Customer_ID"] = targets[customer_id_column]
    report_columns["Customer Name"] = targets[customer_name_column]
    report_columns.update(
        {
            "Cluster": targets["Cluster"],
            "Current Rebate": targets[REBATE_COL],
            "Cluster Median Rebate": targets["cluster_benchmark_cpl"],
            "Rebate Gap": targets["rebate_gap_cpl"],
            "Financial Opportunity": targets["rebate_opportunity_gbp"],
            "Outlier Flag": targets["rebate_flag"],
        }
    )

    report = pd.DataFrame(report_columns)
    return report.round(2)


def filter_by_cluster(df: pd.DataFrame, selected_cluster):
    if selected_cluster == "All":
        return df
    return df[df["Cluster"] == selected_cluster]


def render_metric_cards(df: pd.DataFrame) -> None:
    targets = df[df["rebate_flag"] == "Over Rebated"]
    total_opportunity = targets["rebate_opportunity_gbp"].sum()
    average_gap = targets["rebate_gap_cpl"].mean() if not targets.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Customers",
        f"{len(df):,}",
        help="Total number of customer records included in the current cluster analysis.",
    )
    col2.metric(
        "Renewal Targets",
        f"{len(targets):,}",
        help="Customers flagged as over-rebated because their rebate is more than 1.5 standard deviations above their cluster benchmark.",
    )
    col3.metric(
        "Potential Recovery",
        f"GBP {total_opportunity:,.0f}",
        help="Estimated rebate savings from reducing over-rebated customers to their cluster median rebate level.",
    )
    col4.metric(
        "Average Target Gap",
        f"{average_gap:.2f} cpl",
        help="Average rebate gap in cents per litre across customers flagged as renewal targets.",
    )


def render_boxplots(df: pd.DataFrame) -> None:
    preferred_columns = [
        REBATE_COL,
        "total_vol",
        "total_dpm",
        "national_volume",
        "international_volume",
        "used_retail_sites_count",
        "total_rebate_amount",
    ]
    plot_columns = [
        column
        for column in preferred_columns
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not plot_columns:
        st.info("No numeric columns are available for cluster boxplots.")
        return

    n_cols = 3
    n_rows = math.ceil(len(plot_columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.2 * n_rows))
    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(plot_columns):
        label = DISPLAY_LABELS.get(column, column.replace("_", " ").title())
        sns.boxplot(
            data=df,
            x="Cluster",
            y=column,
            showfliers=False,
            ax=axes[index],
        )
        axes[index].set_title(f"{label} by Cluster", fontsize=11)
        axes[index].set_xlabel("Cluster")
        axes[index].set_ylabel(label)
        axes[index].tick_params(axis="x", labelrotation=0)
        axes[index].tick_params(axis="y", labelsize=9)

    for index in range(len(plot_columns), len(axes)):
        fig.delaxes(axes[index])

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close(fig)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_signed_in_sidebar(
    title: str,
    signed_in_name: str,
    authenticator: stauth.Authenticate,
    logout_key: str,
) -> None:
    st.sidebar.title(title)
    st.sidebar.caption(f"Signed in as {signed_in_name}")
    authenticator.logout("Sign Out", location="sidebar", key=logout_key)


def load_pricing_source_data(uploaded_original_file, uploaded_pca_file) -> pd.DataFrame:
    original_df = load_csv(
        uploaded_original_file,
        load_default_original_data,
        ORIGINAL_DATA_PATH,
    )
    pca_df = load_csv(uploaded_pca_file, load_default_pca_data, PCA_DATA_PATH)
    return combine_original_and_pca_data(original_df, pca_df)


def render_pricing_controls() -> tuple[object, object, int, bool]:
    uploaded_original_file = st.sidebar.file_uploader(
        "Upload Original CSV",
        type=["csv"],
        help=(
            f"Optional. Upload the original unscaled business file. "
            f"If no file is uploaded, the app uses `{ORIGINAL_DATA_PATH}`."
        ),
    )
    uploaded_pca_file = st.sidebar.file_uploader(
        "Upload PCA Output CSV",
        type=["csv"],
        help=(
            "Optional. Upload a prepared PCA output file with PC component columns. "
            f"If no file is uploaded, the app uses `{PCA_DATA_PATH}`."
        ),
    )
    n_clusters = st.sidebar.selectbox("Select Number of Clusters", [3, 4, 5], index=2)
    run_analysis_clicked = st.sidebar.button("Run Analysis", type="primary")
    return uploaded_original_file, uploaded_pca_file, n_clusters, run_analysis_clicked


def get_or_run_pricing_analysis(
    source_df: pd.DataFrame,
    n_clusters: int,
    run_analysis_clicked: bool,
) -> tuple[pd.DataFrame, int, float | None]:
    source_signature = dataframe_signature(source_df)
    analysis_is_stale = (
        "analysis_df" not in st.session_state
        or st.session_state.get("n_clusters") != n_clusters
        or st.session_state.get("source_signature") != source_signature
    )

    if not run_analysis_clicked and "analysis_df" not in st.session_state:
        st.info(
            "Choose a cluster count in the sidebar and run the analysis. "
            f"The app uses `{ORIGINAL_DATA_PATH}` plus `{PCA_DATA_PATH}` by default, "
            "or the uploaded original and PCA output CSV files."
        )
        st.dataframe(source_df.head(25), use_container_width=True)
        st.stop()

    if run_analysis_clicked or analysis_is_stale:
        with st.spinner("Running KMeans clustering and rebate analysis from saved PCA output..."):
            analysis_df, active_silhouette_score = run_analysis(source_df, n_clusters)

        st.session_state["analysis_df"] = analysis_df
        st.session_state["silhouette_score"] = active_silhouette_score
        st.session_state["n_clusters"] = n_clusters
        st.session_state["source_signature"] = source_signature

    return (
        st.session_state["analysis_df"],
        st.session_state["n_clusters"],
        st.session_state.get("silhouette_score"),
    )


def render_cluster_analysis_tab(analysis_df: pd.DataFrame) -> None:
    st.subheader("Cluster Statistics")
    cluster_statistics_df = build_cluster_statistics(analysis_df)
    st.dataframe(
        cluster_statistics_df.style.apply(highlight_min_max, axis=None),
        use_container_width=True,
    )

    st.caption(
        "Cluster labels are generated from saved PC1-PC5 values. "
        "Statistics, rebate analysis, and boxplots use the original unscaled business columns; "
        "PCA values are not used in the cluster summaries."
    )

    render_cluster_characteristics(analysis_df)

    st.subheader("Cluster Metric Boxplots")
    render_boxplots(analysis_df)


def render_opportunity_report_tab(
    analysis_df: pd.DataFrame,
    selected_cluster,
) -> None:
    filtered_analysis_df = filter_by_cluster(analysis_df, selected_cluster)
    target_report = build_target_customer_table(filtered_analysis_df)

    st.subheader("Opportunity Summary")
    st.dataframe(build_opportunity_summary(filtered_analysis_df), use_container_width=True)

    st.subheader("Target Customer Table")
    if target_report.empty:
        st.info("No over-rebated renewal targets found for the selected cluster.")
    else:
        st.dataframe(target_report, use_container_width=True)

    st.download_button(
        label="Download Rebate Opportunity Report",
        data=dataframe_to_csv_bytes(target_report),
        file_name="rebate_opportunity_report.csv",
        mime="text/csv",
        disabled=target_report.empty,
    )


def render_pricing_dashboard(authenticator, signed_in_name: str) -> None:
    st.title("Pricing Team")
    st.subheader("Customer Rebate Optimization Dashboard")
    st.write(
        "This app identifies customers receiving higher rebates than comparable "
        "customers in the same cluster and estimates potential rebate savings during renewal."
    )

    render_signed_in_sidebar(
        "Controls",
        signed_in_name,
        authenticator,
        "pricing_team_logout",
    )
    (
        uploaded_original_file,
        uploaded_pca_file,
        n_clusters,
        run_analysis_clicked,
    ) = render_pricing_controls()

    try:
        source_df = load_pricing_source_data(uploaded_original_file, uploaded_pca_file)
    except FileNotFoundError:
        st.error(
            f"Could not find `{ORIGINAL_DATA_PATH}` or `{PCA_DATA_PATH}`. "
            "Upload compatible files or run `python build_pca_output.py` to regenerate the PCA output."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Could not load or combine the CSV files: {exc}")
        st.stop()

    missing_columns = validate_input(source_df)
    if missing_columns:
        st.error(
            "The dataset is missing required columns: "
            + ", ".join(f"`{column}`" for column in missing_columns)
        )
        st.stop()

    try:
        analysis_df, active_cluster_count, active_silhouette_score = (
            get_or_run_pricing_analysis(source_df, n_clusters, run_analysis_clicked)
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.stop()

    cluster_options = ["All"] + sorted(analysis_df["Cluster"].unique().tolist())
    selected_cluster = st.sidebar.selectbox(
        "Filter Opportunity Report by Cluster",
        cluster_options,
    )

    silhouette_text = ""
    if active_silhouette_score is not None:
        silhouette_text = f" | Silhouette score: {active_silhouette_score:.3f}"
    st.caption(f"Current analysis uses {active_cluster_count} clusters{silhouette_text}.")
    render_metric_cards(analysis_df)

    tab1, tab2 = st.tabs(["Cluster Analysis", "Rebate Opportunity Report"])

    with tab1:
        render_cluster_analysis_tab(analysis_df)

    with tab2:
        render_opportunity_report_tab(analysis_df, selected_cluster)


def render_data_science_dashboard(authenticator, signed_in_name: str) -> None:
    st.title("Data Science Team")
    st.write("Review PCA output and compare clustering model quality.")

    render_signed_in_sidebar(
        "Data Science Controls",
        signed_in_name,
        authenticator,
        "data_science_logout",
    )

    uploaded_pca_file = st.sidebar.file_uploader(
        "Upload PCA Output CSV",
        type=["csv"],
        help=(
            "Optional. Upload a prepared PCA output file with PC component columns. "
            f"If no file is uploaded, the app uses `{PCA_DATA_PATH}`."
        ),
    )
    uploaded_metadata_file = st.sidebar.file_uploader(
        "Upload PCA Metadata CSV",
        type=["csv"],
        help=(
            "Optional. Upload matching PCA metadata with explained variance columns. "
            f"If no custom PCA output is uploaded, the app uses `{PCA_METADATA_PATH}`."
        ),
    )

    try:
        pca_df = load_csv(uploaded_pca_file, load_default_pca_data, PCA_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find `{PCA_DATA_PATH}`. Run `python build_pca_output.py`.")
        st.stop()
    except Exception as exc:
        st.error(f"Could not load PCA output: {exc}")
        st.stop()

    try:
        available_pca_columns = get_available_pca_columns(pca_df)
        if not available_pca_columns:
            raise ValueError("No PCA component columns were found in the PCA output.")
        pca_features = prepare_pca_features(pca_df, tuple(available_pca_columns))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    metadata_df = pd.DataFrame()
    if uploaded_metadata_file is not None:
        try:
            metadata_df = load_uploaded_csv(
                uploaded_metadata_file.name,
                uploaded_metadata_file.getvalue(),
            )
        except Exception as exc:
            st.warning(f"Could not load uploaded PCA metadata: {exc}")
    elif uploaded_pca_file is None:
        try:
            metadata_df = load_default_pca_metadata(*file_cache_key(PCA_METADATA_PATH))
        except FileNotFoundError:
            st.warning(
                f"`{PCA_METADATA_PATH}` was not found. Run `python build_pca_output.py` "
                "to refresh explained variance metadata."
            )
        except Exception as exc:
            st.warning(f"Could not load PCA metadata: {exc}")
    else:
        st.warning(
            "Uploaded PCA output is being used without matching PCA metadata. "
            "Explained variance is unavailable for this uploaded file."
        )

    component_count = len(available_pca_columns)
    if not metadata_df.empty and "Explained Variance Ratio" in metadata_df.columns:
        total_explained_variance = metadata_df["Explained Variance Ratio"].sum()
    else:
        total_explained_variance = np.nan

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Rows", f"{len(pca_df):,}")
    metric_col2.metric("PCA Components", component_count)
    if pd.notna(total_explained_variance):
        metric_col3.metric("Explained Variance", f"{total_explained_variance:.1%}")
    else:
        metric_col3.metric("Explained Variance", "Unavailable")

    if not metadata_df.empty:
        st.subheader("PCA Explained Variance")
        variance_df = metadata_df.copy()
        for column in ["Explained Variance Ratio", "Cumulative Explained Variance"]:
            if column in variance_df.columns:
                variance_df[column] = (variance_df[column] * 100).round(2)
        st.dataframe(variance_df, use_container_width=True)

    st.subheader("Model Comparison")
    pca_signature = (
        dataframe_signature(pca_features),
        KMEDOIDS_SAMPLE_SIZE,
        AGGLOMERATIVE_SAMPLE_SIZE,
    )
    if st.session_state.get("data_science_pca_signature") != pca_signature:
        st.session_state["data_science_results"] = pd.DataFrame()
        st.session_state["data_science_pca_signature"] = pca_signature

    control_col1, control_col2, control_col3 = st.columns([1, 1, 1])
    with control_col1:
        cluster_counts = st.multiselect(
            "Cluster Counts",
            options=list(range(2, 11)),
            default=[3, 4, 5],
            help="Cluster counts to evaluate for each selected model.",
        )
    with control_col2:
        model_names = st.multiselect(
            "Models",
            options=[
                "KMeans",
                "KMedoids",
                "Agglomerative Clustering",
                "Gaussian Mixture",
            ],
            default=["KMeans", "Gaussian Mixture"],
        )
    with control_col3:
        pca_component_count = st.selectbox(
            "PCA Components",
            options=list(range(1, component_count + 1)),
            index=min(4, component_count - 1),
            help="Number of PCA components to include in model comparison.",
        )

    selected_pca_columns = tuple(available_pca_columns[:pca_component_count])

    run_models_clicked = st.button("Run Model Comparison", type="primary")
    clear_results_clicked = st.button("Clear Stored Model Scores")
    if clear_results_clicked:
        st.session_state["data_science_results"] = pd.DataFrame()

    if run_models_clicked:
        if not cluster_counts:
            st.error("Select at least one cluster count.")
            st.stop()
        if not model_names:
            st.error("Select at least one model.")
            st.stop()

        with st.spinner("Running model comparison..."):
            results_df = run_data_science_models(
                pca_features,
                tuple(sorted(cluster_counts)),
                tuple(model_names),
                selected_pca_columns,
                KMEDOIDS_SAMPLE_SIZE,
                AGGLOMERATIVE_SAMPLE_SIZE,
            )
        existing_results = st.session_state.get(
            "data_science_results",
            pd.DataFrame(),
        )
        st.session_state["data_science_results"] = append_model_scores(
            existing_results,
            results_df,
        )

    if not st.session_state.get("data_science_results", pd.DataFrame()).empty:
        st.dataframe(st.session_state["data_science_results"], use_container_width=True)
    else:
        st.info("Select cluster counts and models, then run the comparison.")


authenticator, signed_in_name, signed_in_username = require_login()

if signed_in_username == "data_science":
    render_data_science_dashboard(authenticator, signed_in_name)
    st.stop()

render_pricing_dashboard(authenticator, signed_in_name)
