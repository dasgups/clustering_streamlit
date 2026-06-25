import argparse

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DEFAULT_INPUT_PATH = "segmentation_input_updated.csv"
DEFAULT_OUTPUT_PATH = "pca_5d_output.csv"
DEFAULT_METADATA_PATH = "pca_metadata.csv"
PCA_VARIANCE_TARGET = 0.90
MIN_COMPONENTS = 5

EXCLUDED_CLUSTERING_COLUMNS = [
    "parent_name",
    "supplied_by_bp_median_rebate_cpl",
    "non_supplied_by_bp_median_rebate_cpl",
    "non_street_heavy_sites_median_rebate_cpl",
    "street_heavy_sites_median_rebate_cpl",
    "dealer_heavy_sites_median_rebate_cpl",
    "bp_heavy_sites_median_rebate_cpl",
    "overall_median_rebate_cpl",
    "total_rebate_amount",
]


def signed_log1p(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def prepare_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()

    if "earliest_purchase_month" in model_df.columns:
        model_df["contract_age"] = 13 - pd.to_numeric(
            model_df["earliest_purchase_month"],
            errors="coerce",
        ).fillna(0)
        model_df = model_df.drop(columns=["earliest_purchase_month"])

    model_df = model_df.drop(columns=EXCLUDED_CLUSTERING_COLUMNS, errors="ignore")

    numeric_columns = model_df.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        model_df[column] = signed_log1p(model_df[column])

    model_df = pd.get_dummies(model_df, drop_first=True)
    model_df = model_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return model_df


def select_component_count(
    explained_variance_ratio: np.ndarray,
    variance_target: float = PCA_VARIANCE_TARGET,
    min_components: int = MIN_COMPONENTS,
) -> int:
    cumulative_variance = np.cumsum(explained_variance_ratio)
    target_components = np.searchsorted(cumulative_variance, variance_target) + 1
    return int(max(min_components, target_components))


def build_pca_frames(
    scaled_data: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_pca = PCA(random_state=1)
    full_pca.fit(scaled_data)
    n_components = select_component_count(full_pca.explained_variance_ratio_)

    pca = PCA(n_components=n_components, random_state=1)
    pca_data = pca.fit_transform(scaled_data)
    component_columns = [f"PC{index}" for index in range(1, n_components + 1)]

    pca_df = pd.DataFrame(pca_data, columns=component_columns)
    metadata_df = pd.DataFrame(
        {
            "Component": component_columns,
            "Explained Variance Ratio": pca.explained_variance_ratio_,
            "Cumulative Explained Variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return pca_df, metadata_df


def build_pca_output(input_path: str, output_path: str, metadata_path: str) -> None:
    original_df = pd.read_csv(input_path)
    encoded_df = prepare_clustering_features(original_df)

    if encoded_df.shape[1] < 5:
        raise ValueError(
            "The clustering dataset must contain at least five encoded features for PCA."
        )

    scaled_data = StandardScaler().fit_transform(encoded_df)
    pca_df, metadata_df = build_pca_frames(scaled_data)

    output_df = pd.concat([original_df.reset_index(drop=True), pca_df], axis=1)
    output_df.to_csv(output_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    print(
        f"Wrote {output_path} with {len(output_df):,} rows "
        f"and {len(output_df.columns):,} columns."
    )
    print(
        f"Wrote {metadata_path} with {len(pca_df.columns)} PCA components "
        f"covering {metadata_df['Cumulative Explained Variance'].iloc[-1]:.1%} variance."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the saved five-component PCA output for the Streamlit app."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Raw CSV input path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"PCA output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"PCA metadata CSV path. Default: {DEFAULT_METADATA_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_pca_output(args.input, args.output, args.metadata)
