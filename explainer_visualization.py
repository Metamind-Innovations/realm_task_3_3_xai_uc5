import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import shap
from typing import Tuple
from utils import load_pickle


def shap_visualizations(
    shap_values: shap.Explanation,
    output_dir: Path,
    figsize: Tuple[float, float] = (8, 4),
) -> None:
    """Create and save SHAP visualizations for model interpretability.

    Args:
        shap_values (shap.Explanation): SHAP values object obtained from a SHAP
            explainer.
        output_dir (Path): Directory path where the plot images will be saved.
        figsize (Tuple[float, float], optional): Figure size for both
                                            plots. Defaults to (8, 4).
    """

    # Beeswarm Plot (Summary Plot)
    plt.figure(figsize=figsize)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Bar Plot (Feature Importance)
    plt.figure(figsize=figsize)
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_explanations(
    analysis_results: Path, output_dir: Path, method: str
) -> None:
    """Visualize permutation feature importance or counterfactual explanations and save plots.
    This function loads analysis results, checks from which methos the results have beed
    generated and creates visualizations.

    Args:
        analysis_results (str or Path): Path to the file containing analysis results.
        output_dir (str or Path): Directory where the generated plots will be saved.
        method (str): XAI method used.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "shap":
        shap_values = load_pickle(analysis_results)
        shap_visualizations(shap_values=shap_values, output_dir=output_dir)

    elif method == "lime":
        pass

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
        "--method",
        type=str,
        choices=["lime", "shap"],
        required=True,
        help="Explainability method: lime or shap",
    )

    args = parser.parse_args()

    visualize_explanations(
        analysis_results=Path(args.analysis_results),
        output_dir=Path(args.output),
        method=args.method,
    )


if __name__ == "__main__":
    main()
