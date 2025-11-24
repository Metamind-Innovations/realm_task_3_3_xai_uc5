import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import shap
from typing import Tuple, Dict
from utils import load_pickle, load_json, select_method


def shap_visualizations(
    shap_values: shap.Explanation,
    output_dir: Path,
    sensitivity: float,
    figsize: Tuple[float, float] = (8, 4),
) -> None:
    """Create and save SHAP visualizations for model interpretability.

    Args:
        shap_values (shap.Explanation): SHAP values object obtained from a SHAP
            explainer.
        output_dir (Path): Directory path where the plot images will be saved.
        sensitivity (float): Sensitivity value used for analysis.
        figsize (Tuple[float, float], optional): Figure size for both
                                            plots. Defaults to (8, 4).
    """

    # Beeswarm Plot (Summary Plot)
    plt.figure(figsize=figsize)
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(
        f"Sensitivity [0,1]: {sensitivity}, Methodology: SHAP",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    explanation_text = (
        "Shows how each feature impacts model predictions across all samples. "
        "Color indicates feature value (red=high, blue=low). "
        "Horizontal position shows feature impact on each prediction. "
        "Features ordered vertically by importance (top features most important)."
    )
    plt.figtext(
        0.5,
        0.01,
        explanation_text,
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(output_dir / "shap_beeswarm_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar Plot (Feature Importance)
    plt.figure(figsize=figsize)
    shap.plots.bar(shap_values, show=False)
    plt.title(
        f"Sensitivity [0,1]: {sensitivity}, Methodology: SHAP",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    explanation_text = (
        "Average absolute SHAP values showing overall feature importance. "
        "Higher values indicate features with greater impact on model predictions. "
        "Features ranked from most to least important."
    )
    plt.figtext(
        0.5,
        0.02,
        explanation_text,
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_dir / "shap_bar_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def lime_visualization(
    lime_importance_values: Dict[str, Dict[str, float]],
    output_dir: Path,
    sensitivity: float,
    figsize: Tuple[float, float] = (8, 4),
) -> None:
    """Create and save LIME visualization for model interpretability.

    Args:
        lime_importance_values (Dict[str, Dict[str, float]]): Dictionary containing
            LIME importance values for each feature.
        output_dir (Path): Directory path where the plot image will be saved.
        sensitivity (float): Sensitivity value used for analysis.
        figsize (Tuple[float, float], optional): Figure size. Defaults to (8, 4).
    """

    sorted_pairs = sorted(
        lime_importance_values.items(), key=lambda x: x[1]["importance"], reverse=True
    )

    features = [pair[0] for pair in sorted_pairs]
    importance_values = [pair[1]["importance"] for pair in sorted_pairs]

    # Bar Plot (Feature Importance)
    plt.figure(figsize=figsize)

    bars = plt.barh(features, importance_values, color="steelblue")
    plt.gca().invert_yaxis()

    plt.title(
        f"Sensitivity [0,1]: {sensitivity}, Methodology: LIME",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    xmax = max(importance_values) * 1.12
    plt.xlim(0, xmax)

    for bar, val in zip(bars, importance_values):
        plt.text(
            val + (xmax * 0.01),  # slight offset to the right of the bar
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
        )

    explanation_text = (
        "Aggregated feature importance across multiple local explanations. "
        "Higher values indicate features that consistently influence predictions. "
        "Features ordered by decreasing importance from top to bottom."
    )
    plt.figtext(
        0.5,
        0.02,
        explanation_text,
        ha="center",
        fontsize=10,
        style="italic",
        color="#555555",
        wrap=True,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    plt.savefig(output_dir / "lime_barplot.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_explanations(
    analysis_results: Path, output_dir: Path, sensitivity: float
) -> None:
    """Visualize permutation feature importance or counterfactual explanations and save plots.
    This function loads analysis results, checks from which method the results have beed
    generated and creates visualizations.

    Args:
        analysis_results (str or Path): Path to the file containing analysis results.
        output_dir (str or Path): Directory where the generated plots will be saved.
        sensitivity (float): Sensitivity for XAI method used.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find XAI method used
    method = select_method(sensitivity)

    if method == "shap":
        shap_values = load_pickle(analysis_results)
        shap_visualizations(
            shap_values=shap_values, output_dir=output_dir, sensitivity=sensitivity
        )

    elif method == "lime":
        lime_importance_values = load_json(analysis_results)
        lime_visualization(
            lime_importance_values=lime_importance_values,
            output_dir=output_dir,
            sensitivity=sensitivity,
        )

    print(f"Plots stored in {output_dir}")


def main() -> None:
    """Entry point for visualizing explainability analysis results.
    Parses command-line arguments to specify the analysis results file,
    the output directory, and the xai method, then calls
    `visualize_explanations` to generate and save the plots.
    """

    parser = argparse.ArgumentParser(
        description="Visualize explainability analysis results for COPowereDWrapper model predictions"
    )
    parser.add_argument(
        "--analysis_results",
        help="Analysis file path",
    )
    parser.add_argument(
        "--output", default="output", help="Output dir for visualizations"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.7,
        help="Sensitivity value [0-1]: <0.5 for lime, >=0.5 for shap",
    )

    args = parser.parse_args()

    visualize_explanations(
        analysis_results=Path(args.analysis_results),
        output_dir=Path(args.output),
        sensitivity=args.sensitivity,
    )


if __name__ == "__main__":
    main()
