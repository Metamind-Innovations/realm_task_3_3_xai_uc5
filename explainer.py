import argparse
from typing import Dict, List, Any, Tuple, Union, DefaultDict
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shap
import numpy as np
import lime
from tqdm import tqdm
from collections import defaultdict
import re

from utils import load_csv, store_json, is_between, save_pickle, select_method
from COPowereD_model import COPowereDWrapper


RANDOM_STATE = 2025
TARGET_COLUMN = "label"

CATEGORICAL_COLUMNS = [
    "Gender",
    "b_COPD",
    "s_Worsening",
    "s_Breath",
    "s_Cough",
    "s_Sputum",
]
NUMERICAL_COLUMNS = [
    "Age",
    "Height",
    "Weight",
    "BMI",
    "b_HeartRate",
    "b_SPO2",
    "c_HeartRate",
    "c_SPO2",
]


def pass_checks(
    tabular_data: pd.DataFrame,
) -> bool:
    """Validate that input DataFrame passes basic checks.

    Args:
        tabular_data (pd.DataFrame): DataFrame containing patient tabular data with
            target labels.

    Returns:
        bool: True if all validation checks pass, False otherwise.
    """

    if tabular_data is None:
        return False

    if tabular_data.empty:
        return False

    if TARGET_COLUMN not in tabular_data.columns:
        return False

    return True


def data_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a DataFrame.
    This function fills missing values in numerical columns with the median,
    and in categorical columns with the mode (most frequent value).

    Args:
        data (pd.DataFrame): Input DataFrame containing numerical and categorical columns.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """

    return data.apply(
        lambda col: (
            col.fillna(col.median())
            if col.name in NUMERICAL_COLUMNS
            else col.fillna(col.mode().iloc[0])
        ),
        axis=0,
    )


def prepare_data_for_analysis(
    tabular_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare feature and target data from full dataset.

    Args:
        tabular_data (pd.DataFrame): DataFrame containing patient tabular
                                        data actual labels.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - features_df (pd.DataFrame): DataFrame with features.
            - target_df (pd.DataFrame): DataFrame with target values.
            - id_df (pd.DataFrame): DataFrame with id.
    """

    features_df = tabular_data.copy().drop(columns=[TARGET_COLUMN, "id"])

    target_df = tabular_data[[TARGET_COLUMN]].copy()
    id_df = tabular_data[["id"]].copy()

    return features_df, target_df, id_df


def stratified_split(
    X: pd.DataFrame, y: pd.DataFrame, background_size=0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified split of dataset.

    Args:
        X (pd.DataFrame): Feature matrix containing patient data.
        y (pd.Series): Target labels corresponding to the features in X.
        background_size (float, optional): Proportion of data to use as
            background for SHAP. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - X_background (pd.DataFrame): Feature matrix for background set.
            - X_explain (pd.DataFrame): Feature matrix for samples to explain.
            - y_background (pd.Series): Labels for background set.
            - y_explain (pd.Series): Labels for samples to explain.
    """

    X_explain, X_background, y_explain, y_background = train_test_split(
        X, y, test_size=background_size, stratify=y, random_state=RANDOM_STATE
    )

    return X_background, X_explain, y_background, y_explain


def shap_to_json(
    shap_values: np.ndarray, feature_names: List[str]
) -> Dict[str, Union[List[float], List[List[float]], List[str]]]:
    """Calculates global SHAP statistics and packages them with individual
    SHAP values into a dictionary suitable for JSON serialization.

    Args:
        shap_values (np.ndarray): SHAP values array of shape (n_samples, n_features).
            Each value represents a feature's contribution to a prediction.
        feature_names (List[str]): List of feature names corresponding to columns
            in shap_values.

    Returns:
        Dict[str, Union[List[float], List[List[float]], List[str]]]: Dictionary containing
                        the summarized results.
    """

    # Calculate global statistics
    shap_abs_mean = np.abs(shap_values).mean(axis=0).tolist()
    shap_direction_mean = shap_values.mean(axis=0).tolist()
    shap_std = shap_values.std(axis=0).tolist()

    shap_values_list = shap_values.tolist()

    summary = {
        "feature_importance": shap_abs_mean,
        "shap_direction_mean": shap_direction_mean,
        "shap_std": shap_std,
        "shap_values": shap_values_list,
        "features": feature_names,
    }

    return summary


def shap_analysis(
    X: pd.DataFrame, y: pd.DataFrame, feature_names: List[str]
) -> Tuple[
    Dict[str, Union[List[float], List[List[float]], List[str]]], shap.Explanation
]:
    """
    Perform SHAP explainability analysis on model predictions.
    Uses Kernel SHAP to explain model predictions by calculating feature
    contributions. Splits data into background (for explainer) and
    explanation (samples to explain) sets using stratified sampling.

    Args:
        X (pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (pd.DataFrame): Target labels.
        feature_names (List[str]): List of feature names corresponding to X columns.

    Returns:
        Tuple[Dict[str, Union[List[float], List[List[float]], List[str]]], shap.Explanation]:
            - summary (dict): Dictionary with SHAP statistics and values.
            - shap_values (shap.Explanation): SHAP Explanation object containing full
                SHAP analysis results.
    """

    # Split data
    X_background, X_explain, y_background, y_explain = stratified_split(X=X, y=y)

    model = COPowereDWrapper(feature_names=feature_names, one_dim_preds=True)

    # SHAP XAI analysis
    shap_explainer = shap.KernelExplainer(model=model.predict_proba, data=X_background)
    shap_values = shap_explainer(X=X_explain)

    # Retrieve results
    summary = shap_to_json(shap_values=shap_values.values, feature_names=feature_names)

    return summary, shap_values


def lime_global_importance(
    weights_per_feature: DefaultDict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Calculate global feature importance metrics from LIME explanations.
    Computes three key metrics for each feature across all instances:
    - Importance: Mean absolute weight (magnitude of influence)
    - Mean weight: Average directional weight (positive/negative influence)
    - Standard deviation: Variability of weights across instances

    Args:
        weights_per_feature (DefaultDict[str, List[float]]): Dictionary
                            mapping feature names to lists of weights.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping feature names to their global importance metrics.
    """

    global_feature_importance = {}

    for feature, weights in weights_per_feature.items():

        global_feature_importance[feature] = {
            "importance": float(np.mean(np.abs(weights))),
            "mean_weight": float(np.mean(weights)),
            "std_weight": float(np.std(weights)),
        }

    return global_feature_importance


def lime_analysis(
    X: pd.DataFrame, y: pd.DataFrame, feature_names: List[str]
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Union[int, float]]]]:
    """Perform LIME explainability analysis on tabular data.
    Generates local explanations for each instance using LIME (Local Interpretable
    Model-agnostic Explanations) and aggregates them into global feature importance
    metrics. Each instance's prediction is explained by computing feature weights
    that indicate how much each feature contributes to the model's prediction.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.DataFrame): Target labels indicating the true class for each sample.
        feature_names (List[str]): List of feature names corresponding to columns in X.

    Returns:
        Tuple containing:
        - Dict[str, Dict[str, float]]: Dictionary mapping each feature to its global
            importance metrics across all instances.
        - List[Dict[str, Union[int, float]]]: List of dictionaries, one per instance, containing
            the feature weights for that specific prediction.
    """

    model = COPowereDWrapper(feature_names=feature_names)

    class_names = list(model.get_class_names().values())
    categorical_indices = [X.columns.get_loc(col) for col in CATEGORICAL_COLUMNS]

    # Data dfs to arrays
    X = X.values
    y = y.values

    # LIME XAI analysis
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        categorical_features=categorical_indices,
        class_names=class_names,
        mode="classification",
    )

    instance_feature_weights = []
    weights_per_feature = defaultdict(list)

    for k in tqdm(range(len(X)), desc="LIME XAI analysis for instance"):

        instance_explanation = lime_explainer.explain_instance(
            data_row=X[k],
            predict_fn=model.predict_proba,
            num_features=len(feature_names),
            top_labels=1,
        )

        explained_class = int(instance_explanation.top_labels[0])
        explanation_list = instance_explanation.as_list(label=explained_class)

        # Current instance weights
        instance_weights = {
            "instance_idx": k,
            "true_class": int(y[k][0]),
            "predicted_class": model.predict(X[k : k + 1]),
            "explained_class": explained_class,
        }

        # Extract weight for each feature
        for feature_desc, weight in explanation_list:
            # Get feature name
            match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", feature_desc)
            if match:
                base_feature = match.group(0)
            else:
                base_feature = feature_desc

            # Store for specific instance
            instance_weights[base_feature] = weight

            # Keep for global calculation
            weights_per_feature[base_feature].append(weight)

        instance_feature_weights.append(instance_weights)

    global_feature_importance = lime_global_importance(weights_per_feature)

    return global_feature_importance, instance_feature_weights


def run_explainability_analysis(
    tabular_data: Union[str, Path],
    output_dir: Union[str, Path],
    sensitivity: float,
) -> List[Dict[str, Any]]:
    """
    Runs an explainability analysis on tabular data using either lime
    or shap, depending on the specified sensitivity.
    The function:
        1. Loads tabular data with features and target labels.
        2. Checks data consistency.
        3. Selects the explainability method based on sensitivity.
           - Low sensitivity (<0.5): LIME
           - High sensitivity (>=0.5): SHAP
        4. Configures the API Endpoint model.
        5. Performs explainability analysis.
        6. Stores results.

    Args:
        tabular_data (Union[str, Path]): Path to CSV file containing data.
        output_dir (Path): Directory where results will be stored.
        sensitivity (float): Sensitivity parameter (0 to 1) controlling the method selection.
    """

    # Load data
    tabular_data = load_csv(tabular_data)

    if not pass_checks(tabular_data):
        raise ValueError(f"Inconsistent data provided. Check again the data provided.")

    # Prepare data
    tabular_data = data_imputation(tabular_data)
    tabular_data, label, id = prepare_data_for_analysis(tabular_data=tabular_data)

    method = select_method(sensitivity)
    print(f"Using method: {method} based on sensitivity: {sensitivity}")

    feature_names = tabular_data.columns.to_list()

    # Perform analysis
    if method == "lime":
        results, detailed_results = lime_analysis(
            X=tabular_data, y=label, feature_names=feature_names
        )

    elif method == "shap":
        results, detailed_results = shap_analysis(
            X=tabular_data, y=label, feature_names=feature_names
        )

    # Store results to output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    store_json(data=results, path=output_dir.joinpath(f"{method}_analysis.json"))

    (
        save_pickle(
            data=detailed_results,
            filepath=output_dir.joinpath(f"{method}_detailed_results.pickle"),
        )
        if method == "shap"
        else store_json(
            data=detailed_results,
            path=output_dir.joinpath(f"{method}_detailed_results.json"),
        )
    )

    print(f"Explainability analysis completed. Results saved to {output_dir}")


def main():
    """
    Main entry point for feature importance analysis using LIME or SHAP.
    This function parses command-line arguments and runs explainability analysis
    on tabular data."""

    parser = argparse.ArgumentParser(
        description="Analyze and explain feature importance"
    )

    parser.add_argument(
        "--tabular_data", required=True, help="Path to tabular data CSV file"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output dir for results JSON files",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.7,
        help="Sensitivity value (0-1): <0.5 uses lime, >=0.5 uses shap",
    )
    args = parser.parse_args()

    if not is_between(x=args.sensitivity):
        raise ValueError("Sensitivity must be between 0 and 1")

    run_explainability_analysis(
        tabular_data=args.tabular_data,
        output_dir=Path(args.output),
        sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()
