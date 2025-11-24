import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_json
from typing import Dict, Tuple, List

AGE_COLORS = {
    "<40": "#66C2A5",
    "40-54": "#FC8D62",
    "55-69": "#8DA0CB",
    "70+": "#E78AC3",
}

AGE_GROUPS = ["<40", "40-54", "55-69", "70+"]

GENDER_COLORS = {"0": "#FF69B4", "1": "#4169E1"}


def demographic_names(data: Dict) -> List[Tuple[str, str]]:
    """
    Extract all available demographic names from a fairness JSON.

    Parameters:
        data (dict): Results JSON containing 'equalized_odds_metrics'
                     and 'demographic_parity_metrics'.

    Returns:
        List: List of demographic names.
    """

    demographic_names = list(data.get("equalized_odds_metrics").keys())

    return demographic_names


def format_method_name(method_name: str) -> str:
    """Format method name by capitalizing and removing underscores.

    Parameters:
        method_name (str): Method name with underscores (e.g., 'equalized_odds').

    Returns:
        str: Formatted method name (e.g., 'Equalized Odds').
    """
    return method_name.replace("_", " ").title()


def plot_consolidated_chart(data: Dict, demographic_name: str, output_dir: Path):
    """
    Create consolidated 1x2 bar chart for fairness and bias metrics.

    Parameters:
        data (dict): Full analysis results
        demographic_name (str): Demographic to visualize (e.g., 'Age', 'Gender')
        output_dir (Path): Directory to save the plot
    """

    fairness_method = "equalized_odds"
    bias_method = "demographic_parity"
    fairness_method_display = format_method_name(fairness_method)
    bias_method_display = format_method_name(bias_method)

    # Determine color palette
    color_map = AGE_COLORS if demographic_name == "Age" else GENDER_COLORS

    # Prepare data structures
    fpr_categories = []
    fpr_values = []

    pred_categories = []
    pred_values = []

    # Extract FPR data (Equalized Odds)
    if demographic_name in data.get("equalized_odds_metrics", {}):
        error_rates = data["equalized_odds_metrics"][demographic_name].get(
            "error_rates_by_group", {}
        )
        for categ, val in error_rates.items():
            fpr_categories.append(categ)
            fpr_values.append(val.get("false_positive_rate"))

    # Extract Prediction Rate data (Demographic Parity)
    if demographic_name in data.get("demographic_parity_metrics", {}):
        pred_rates = data["demographic_parity_metrics"][demographic_name].get(
            "prediction_rates_by_group", {}
        )
        for categ, val in pred_rates.items():
            pred_categories.append(categ)
            pred_values.append(val)

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Fairness and Bias Analysis by {demographic_name}",
        fontsize=16,
        fontweight="bold",
    )

    # Left Plot: False Positive Rate (Fairness)
    if fpr_values:
        fpr_colors = [color_map.get(cat, "#4ECDC4") for cat in fpr_categories]
        bars = axes[0].bar(fpr_categories, fpr_values, color=fpr_colors)

        axes[0].set_title(
            f"Fairness Calculation using {fairness_method_display}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel(demographic_name)
        axes[0].set_ylabel("False Positive Rate")

        ymax = max(fpr_values) * 1.1 if fpr_values else 1
        axes[0].set_ylim(0, ymax)

        # Annotate bars with values
        for bar, val in zip(bars, fpr_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                val + (ymax * 0.01),
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

    # Right Plot: Prediction Rate (Bias)
    if pred_values:
        pred_colors = [color_map.get(cat, "#4ECDC4") for cat in pred_categories]
        bars = axes[1].bar(pred_categories, pred_values, color=pred_colors)

        axes[1].set_title(
            f"Bias Calculation using {bias_method_display}",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel(demographic_name)
        axes[1].set_ylabel("Prediction Rate")

        ymax = max(pred_values) * 1.1 if pred_values else 1
        axes[1].set_ylim(0, ymax)

        # Annotate bars with values
        for bar, val in zip(bars, pred_values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                val + (ymax * 0.01),
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

    # Add explanatory notes
    fig.text(
        0.25,
        0.02,
        "Shows false positive rates across demographic groups. Lower is better.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    fig.text(
        0.75,
        0.02,
        "Shows how often the model predicts positive outcomes. Similar values across groups indicate less bias.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save plot
    filename = f"{demographic_name.lower()}_fairness_bias.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_fairness_bias_analysis(analysis_results: Path, output_dir: Path):
    """
    Create bar charts for all fairness/bias metrics in the JSON results.

    Parameters:
        analysis_results (Path): Path to the JSON file with results
        output_dir (Path): Directory to save the plots
    """

    # load data
    analysis_results = load_json(analysis_results)

    # create dir to store plots
    output_dir.mkdir(parents=True, exist_ok=True)

    # search for demographic names
    demogr_names = demographic_names(analysis_results)

    # Create consolidated plot for each demographic
    for dem_name in demogr_names:
        plot_consolidated_chart(
            data=analysis_results, demographic_name=dem_name, output_dir=output_dir
        )

    print(f"Plots stored in {output_dir}")


def main():
    """
    Main entry point for visualizing fairness and bias analysis results.

    Parses command-line arguments for the JSON file containing metrics
    and the output directory to save the visualizations.
    """

    parser = argparse.ArgumentParser(
        description="Visualize fairness and bias analysis results for COPowereD model predictions"
    )
    parser.add_argument(
        "--analysis_results",
        required=True,
        help="Analysis JSON file path",
    )
    parser.add_argument(
        "--output", default="output", help="Output dir for visualizations"
    )

    args = parser.parse_args()

    visualize_fairness_bias_analysis(
        analysis_results=Path(args.analysis_results), output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()
