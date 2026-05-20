import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Literal, Dict, Tuple, Any, Union
from pathlib import Path
from utils import load_csv, store_json, concat_dfs

ANALYSIS_MAP = {
    "equalized_odds": "error_rates_by_group",
    "demographic_parity": "prediction_rates_by_group",
}
TARGET_COLUMN = "label"
DEMOGRAPHICS_COLUMNS = {"age": "age", "gender": "sexe"}


def pass_checks(
    tabular_data: pd.DataFrame,
    pred_target: pd.DataFrame,
) -> bool:
    """Validate that input DataFrames and required columns pass basic checks.

    Args:
        tabular_data (pd.DataFrame): DataFrame containing patient tabular data with
            demographics and target labels.
        pred_target (pd.DataFrame): DataFrame containing predicted target labels.

    Returns:
        bool: True if all validation checks pass, False otherwise.
    """

    if (tabular_data is None) or (pred_target is None):
        return False

    if tabular_data.empty or pred_target.empty:
        return False

    if not (len(tabular_data) == len(pred_target)):
        return False

    if (TARGET_COLUMN not in tabular_data.columns) or (
        TARGET_COLUMN not in pred_target.columns
    ):
        return False

    if not any(col in tabular_data.columns for col in DEMOGRAPHICS_COLUMNS.values()):
        return False

    return True


def prepare_data_for_analysis(
    tabular_data: pd.DataFrame,
    pred_target: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare demographic, actual target, and predicted target DataFrames for analysis.

    Args:
        tabular_data (pd.DataFrame): DataFrame containing patient tabular data with
            demographics and actual labels.
        pred_target (pd.DataFrame): DataFrame containing predicted target labels.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing three DataFrames:
            - dem_df (pd.DataFrame): DataFrame with demographic columns.
            - actual_df (pd.DataFrame): DataFrame with actual target values renamed to 'actual'.
            - pred_df (pd.DataFrame): DataFrame with predicted target values renamed to 'pred'.
    """

    dem_df = tabular_data[
        [col for col in DEMOGRAPHICS_COLUMNS.values() if col in tabular_data.columns]
    ]

    actual_df = tabular_data[[TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "actual"})
    pred_df = pred_target[[TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "pred"})

    return dem_df, actual_df, pred_df


def age_to_cat(age_col: pd.Series) -> pd.Series:
    """Convert a numeric age column into categorical age groups.

    Age bins: <40, 40-54, 55-69, 70+

    Args:
        age_col (pd.Series): Numeric age values.

    Returns:
        pd.Series: A categorical series with age groups as labels.
    """

    age_bins = [0, 40, 55, 70, 120]
    age_labels = ["<40", "40-54", "55-69", "70+"]

    return pd.cut(
        age_col, bins=age_bins, labels=age_labels, include_lowest=True, right=True
    )


def compute_fpr(true: pd.Series, pred: pd.Series) -> float:
    """Compute False Positive Rate (FPR) for binary classification.

    Args:
        true (pd.Series): Ground truth binary labels (0/1).
        pred (pd.Series): Predicted binary labels (0/1).

    Returns:
        float: False Positive Rate = FP / (FP + TN)
    """

    cm = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return fpr


def calculate_metric(
    data: pd.DataFrame,
    analysis: Literal["equalized_odds", "demographic_parity"],
) -> Dict[str, Any]:
    """Calculate fairness metrics (Equalized Odds or Demographic Parity) per demographic group.

    Args:
        data (pd.DataFrame): DataFrame containing demographic columns,
            actual labels ('actual'), and predicted labels ('pred').
        analysis (Literal["equalized_odds", "demographic_parity"]): Type of fairness
            analysis to perform. Must be either 'equalized_odds' or 'demographic_parity'.

    Returns:
        Dict[str, Any]: A nested dictionary with the analysis results.
    """

    metrics = {}

    for col in DEMOGRAPHICS_COLUMNS.values():
        if col not in data.columns:
            continue

        metrics[col] = {ANALYSIS_MAP.get(analysis): {}}

        unique_vals = data[col].unique().tolist()

        for val in unique_vals:
            if pd.isna(val):
                continue

            spec_cat_data = data[data[col] == val].copy()

            # Check which analysis is requested
            if analysis == "equalized_odds":
                fpr = compute_fpr(
                    true=spec_cat_data["actual"], pred=spec_cat_data["pred"]
                )
                analysis_results = {
                    "false_positive_rate": fpr,
                }
            elif analysis == "demographic_parity":
                analysis_results = spec_cat_data["pred"].mean()

            metrics[col][ANALYSIS_MAP.get(analysis)][f"{val}"] = analysis_results

    return metrics


def fairness_bias_analysis(
    tabular_data: Union[str, Path],
    pred_target: Union[str, Path],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Run fairness and bias analysis using Equalized Odds and Demographic Parity.
    This function loads tabular data with actual labels, and predicted target labels,
    validates them, prepares them for analysis, and computes fairness metrics across
    demographic subgroups. The results are then stored as a JSON file.

    Args:
        tabular_data (Union[str, Path]): Path to the tabular dataset CSV file containing
            patient data and actual labels.
        pred_target (Union[str, Path]): Path to CSV file containing predicted target labels.
        output_path (Union[str, Path]): Path where the results JSON file will be saved.

    Returns:
        Dict[str, Any]: A dictionary containing metrics.

    Raises:
        ValueError: If the provided data is inconsistent (fails validation checks).
    """

    # Init results
    results = {"equalized_odds_metrics": {}, "demographic_parity_metrics": {}}

    # Load data
    tabular_data = load_csv(tabular_data)
    pred_target = load_csv(pred_target)

    if not pass_checks(tabular_data=tabular_data, pred_target=pred_target):
        raise ValueError(f"Inconsistent data provided. Check again the data provided.")

    tabular_data, actual_target, pred_target = prepare_data_for_analysis(
        tabular_data=tabular_data,
        pred_target=pred_target,
    )

    # Convert age columns to categorical
    if DEMOGRAPHICS_COLUMNS.get("age") in tabular_data.columns:
        tabular_data[DEMOGRAPHICS_COLUMNS.get("age")] = age_to_cat(
            tabular_data[DEMOGRAPHICS_COLUMNS.get("age")]
        )

    # Concatenated df with demographic columns and 'actual', 'pred' columns
    concat_data = concat_dfs([tabular_data, actual_target, pred_target])

    # Equalized odds
    results["equalized_odds_metrics"] = calculate_metric(
        data=concat_data, analysis="equalized_odds"
    )

    # Demographic parity
    results["demographic_parity_metrics"] = calculate_metric(
        data=concat_data,
        analysis="demographic_parity",
    )

    # Store results
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    store_json(data=results, path=output_path)
    print(f"Fairness - Bias analysis completed. Results saved to {output_path}")


def main() -> None:
    """CLI entry point for fairness and bias analysis of COPowereD model predictions.
    This function parses command-line arguments for tabular data, predicted
    targets, and output path. It then runs the fairness and bias analysis pipeline.
    """

    parser = argparse.ArgumentParser(
        description="Analyze fairness and bias in COPowereD model predictions"
    )
    parser.add_argument(
        "--tabular_data", required=True, help="Path to tabular data CSV file"
    )
    parser.add_argument(
        "--pred_target", required=True, help="Path to predicted target data CSV file"
    )
    parser.add_argument(
        "--output",
        default="output/fairness_analysis.json",
        help="Output JSON file path",
    )

    args = parser.parse_args()

    fairness_bias_analysis(
        tabular_data=Path(args.tabular_data),
        pred_target=Path(args.pred_target),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
