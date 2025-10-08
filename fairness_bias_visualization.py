import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from utils import load_json
from typing import Dict, Tuple, List


def is_equalized_odds(x: str) -> bool:
    """
    Check if a given metric belongs to the Equalized Odds.

    Parameters:
        metric_name (str): The metric name to check

    Returns:
        bool: True if the metric is part of Equalized Odds, False otherwise
    """
    return True if x in ["false_positive_rate"] else False


def is_demographic_parity(x: str) -> bool:
    """
    Check if a given metric belongs to the Demographic Parity.

    Parameters:
        metric_name (str): The metric name to check

    Returns:
        bool: True if the metric is part of Demographic Parity, False otherwise
    """
    return True if x in ["prediction_rates_by_group"] else False


def demographics_metrics_pairs(data: Dict) -> List[Tuple[str, str]]:
    """
    Extract all available (demographic, metric) pairs from a fairness JSON.

    Parameters:
        data (dict): Results JSON containing 'equalized_odds_metrics'
                     and 'demographic_parity_metrics'.

    Returns:
        List[Tuple[str, str]]: List of (demographic_name, metric_name) pairs
                               that exist in the JSON.
    """

    demographic_names = list(data.get("equalized_odds_metrics").keys())

    available_metrics = ["false_positive_rate", "prediction_rates_by_group"]

    demogr_metrics = [(x, y) for x in demographic_names for y in available_metrics]

    return demogr_metrics


def plot_bar_chart(
    x, y, title="", xlabel="", ylabel="", figsize=(6, 4), output_path=None, show=False
):
    """
    Bar chart function.

    Parameters:
        x (list): Categories / groups
        y (list): Values corresponding to x
        title (str): Plot title
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (tuple, optional): Figure size (default: (6, 4))
        output_path (str or Path, optional): If provided, save the plot to this path
        show (bool, optional): Whether to display the plot (default: True)
    """

    plt.figure(figsize=figsize)
    bars = plt.bar(x, y, color="steelblue")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add padding above bars
    ymax = max(y) * 1.1 if y else 1
    plt.ylim(0, ymax)

    # Annotate bars with values
    for bar, val in zip(bars, y):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + (ymax * 0.01),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
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

    # search for demographics - metrics pairs
    demogr_metr = demographics_metrics_pairs(analysis_results)

    # get categories, values and plot
    for dem_name, metric in demogr_metr:
        categories, values = [], []

        if is_equalized_odds(metric):
            metric_group = "equalized_odds_metrics"
            rates = "error_rates_by_group"
        elif is_demographic_parity(metric):
            metric_group = "demographic_parity_metrics"
            rates = "prediction_rates_by_group"

        for categ, val in (
            analysis_results.get(metric_group).get(dem_name).get(rates).items()
        ):
            categories.append(categ)
            values.append(val.get(metric) if is_equalized_odds(metric) else val)

        # create and store plot
        plot_bar_chart(
            x=categories,
            y=values,
            title=f"{metric} ({dem_name} groups)",
            xlabel=dem_name,
            ylabel=metric,
            output_path=output_dir.joinpath(f"{dem_name}_{metric}.png"),
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
