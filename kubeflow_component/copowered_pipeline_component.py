from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

# Insert your dockerhub image below (e.g. "docker.io/<username>/<image_name>:<tag>")
DOCKER_IMAGE = "<docker_image>"
PYTHON_BASE_IMAGE = "python:3.14-slim"


@dsl.component(base_image=PYTHON_BASE_IMAGE)
def download_repo(
    github_repo_url: str,
    project_files: Output[Model],
    data: Output[Dataset],
    branch: str = "main",
) -> None:
    """Download project scripts and data from a GitHub repository.

    :param github_repo_url: URL of the GitHub repository to clone.
    :param project_files: Output path for project scripts.
    :param data: Output path for data folder.
    :param branch: Branch name to pull from.
    """
    import shutil
    import subprocess
    from pathlib import Path

    repo_dir = Path("/tmp/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "git"], check=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            github_repo_url,
            str(repo_dir),
        ],
        check=True,
    )
    print(f"Cloned repo {github_repo_url} (branch: {branch}).")

    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)

    for item in repo_dir.iterdir():
        if item.is_file() and item.suffix == ".py":
            shutil.copy2(item, proj_path / item.name)
            print(f"Copied {item.name}")

    required_files = [
        "COPowereD_model.py",
        "explainer.py",
        "fairness_bias_analysis.py",
        "fairness_bias_visualization.py",
        "explainer_visualization.py",
        "utils.py",
    ]
    missing_files = [
        filename for filename in required_files if not (proj_path / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"

    if not src_data_path.exists():
        raise FileNotFoundError("The downloaded repository does not contain data/.")

    for item in src_data_path.iterdir():
        if item.is_file():
            shutil.copy2(item, data_path / item.name)
            print(f"Copied data/{item.name}")
        elif item.is_dir():
            shutil.copytree(item, data_path / item.name, dirs_exist_ok=True)
            print(f"Copied data/{item.name}/ directory")

    data_csv = data_path / "data.csv"
    if not data_csv.exists():
        raise FileNotFoundError("Missing required data file: data.csv")

    with data_csv.open("r", encoding="utf-8-sig", newline="") as csv_file:
        header = csv_file.readline().strip().split(",")

    if "label" not in header:
        raise ValueError("The data/data.csv file must contain a 'label' column.")


@dsl.container_component
def copowered_predictions(
    data: Input[Dataset],
    predictions: Output[Dataset],
):
    """Run COPowereD model predictions using the Docker image.

    :param data: Input dataset path containing labeled ``data.csv``.
    :param predictions: Output path for ``result.csv`` with a ``proba`` column.
    """

    # Find and remove label column, and execute the docker model
    command_str = f"""
        set -e
        mkdir -p /app/data/data
        mkdir -p /app/data/result
        mkdir -p "{predictions.path}"
        awk 'BEGIN {{ FS = OFS = "," }}
            {{
                sub(/\\r$/, "", $NF)
                if (NR == 1) {{
                    label_col = 0
                    for (i = 1; i <= NF; i++) {{
                        if ($i == "label") {{
                            label_col = i
                        }}
                    }}
                }}
                line = ""
                for (i = 1; i <= NF; i++) {{
                    if (i != label_col) {{
                        line = line (line == "" ? "" : OFS) $i
                    }}
                }}
                print line
            }}' "{data.path}/data.csv" > /app/data/data/data.csv
        test -s /app/data/data/data.csv
        cd /app
        printf '%s\\n' \
            'log4j.rootLogger=ERROR, stdout' \
            'log4j.appender.stdout=org.apache.log4j.ConsoleAppender' \
            'log4j.appender.stdout.Target=System.out' \
            'log4j.appender.stdout.layout=org.apache.log4j.PatternLayout' \
            > log4j.properties
        java -cp /app/sparkServer-assembly-1.2.0.jar MLProjects.bpco.triage.models.CopdComunicare_001 \
            --input_folder /app/data/data \
            --output_folder /app/data/result
        test -f /app/data/result/result.csv
        cp /app/data/result/result.csv "{predictions.path}/result.csv"
    """

    return dsl.ContainerSpec(
        image=DOCKER_IMAGE,
        command=["sh", "-c"],
        args=[command_str],
    )


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        "numpy>=2.3.3",
        "pandas>=2.3.3",
        "scikit-learn>=1.7.2",
    ],
)
def fairness_analysis(
    project_files: Input[Model],
    data: Input[Dataset],
    predictions: Input[Dataset],
    fairness_results: Output[Dataset],
) -> None:
    """Run fairness and bias analysis for the COPowereD model.

    :param project_files: Input path containing project scripts.
    :param data: Input path containing labeled ``data.csv``.
    :param predictions: Input path containing Docker model ``result.csv``.
    :param fairness_results: Output path for fairness analysis results.
    """
    import subprocess
    from pathlib import Path

    import pandas as pd

    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    predictions_path = Path(predictions.path)
    results_path = Path(fairness_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "fairness_bias_analysis.py"
    result_csv = predictions_path / "result.csv"
    tabular_csv = data_path / "data.csv"

    if not script.exists():
        raise FileNotFoundError(f"Fairness analyzer script not found at {script}")
    if not result_csv.exists():
        raise FileNotFoundError(f"Predictions file not found at {result_csv}")
    if not tabular_csv.exists():
        raise FileNotFoundError(f"Tabular data file not found at {tabular_csv}")

    pred_df = pd.read_csv(result_csv)
    tabular_df = pd.read_csv(tabular_csv)

    if "proba" not in pred_df.columns:
        raise ValueError("Predictions file must contain a 'proba' column.")
    if len(pred_df) != len(tabular_df):
        raise ValueError(
            "Predictions and tabular data must contain the same number of rows."
        )

    pred_csv = results_path / "pred.csv"
    pred_df["label"] = (pred_df["proba"] >= 0.5).astype(int)
    pred_df[["label"]].to_csv(pred_csv, index=False)

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(tabular_csv),
        "--pred_target",
        str(pred_csv),
        "--output",
        str(results_path / "fairness_analysis.json"),
    ]
    subprocess.run(cmd, check=True)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        "numpy>=2.3.3",
        "pandas>=2.3.3",
        "matplotlib>=3.10.6",
    ],
)
def fairness_visualization(
    project_files: Input[Model],
    fairness_results: Input[Dataset],
    fairness_plots: Output[Dataset],
) -> None:
    """Create visualizations for fairness and bias analysis results.

    :param project_files: Input path containing project scripts.
    :param fairness_results: Input path containing ``fairness_analysis.json``.
    :param fairness_plots: Output path for visualization PNG files.
    """
    import subprocess
    from pathlib import Path

    proj_path = Path(project_files.path)
    results_path = Path(fairness_results.path)
    plots_path = Path(fairness_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "fairness_bias_visualization.py"
    analysis_results_file = results_path / "fairness_analysis.json"

    if not script.exists():
        raise FileNotFoundError(f"Fairness visualization script not found at {script}")
    if not analysis_results_file.exists():
        raise FileNotFoundError(
            f"Fairness analysis results not found at {analysis_results_file}"
        )

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_results_file),
        "--output",
        str(plots_path),
    ]
    subprocess.run(cmd, check=True)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        "numpy>=2.3.3",
        "pandas>=2.3.3",
        "scikit-learn>=1.7.2",
        "shap>=0.48.0",
        "lime>=0.2.0.1",
        "tqdm>=4.67.1",
    ],
)
def explainer_analysis(
    project_files: Input[Model],
    data: Input[Dataset],
    predictions: Input[Dataset],
    explainer_results: Output[Dataset],
    sensitivity: float,
) -> None:
    """Run explainer analysis on the COPowereD model.

    :param project_files: Input path containing project scripts.
    :param data: Input path containing labeled ``data.csv``.
    :param predictions: Input path containing Docker model ``result.csv``.
    :param explainer_results: Output path for explainer results.
    :param sensitivity: Sensitivity parameter for the explainer script.
    """
    import subprocess
    from pathlib import Path

    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    predictions_path = Path(predictions.path)
    results_path = Path(explainer_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "explainer.py"
    tabular_csv = data_path / "data.csv"
    result_csv = predictions_path / "result.csv"

    if not script.exists():
        raise FileNotFoundError(f"Explainer script not found at {script}")
    if not tabular_csv.exists():
        raise FileNotFoundError(f"Tabular data file not found at {tabular_csv}")
    if not result_csv.exists():
        raise FileNotFoundError(f"Predictions file not found at {result_csv}")

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(tabular_csv),
        "--sensitivity",
        str(sensitivity),
        "--output",
        str(results_path),
        "--predictions",
        str(result_csv),
    ]
    subprocess.run(cmd, check=True)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        "numpy>=2.3.3",
        "pandas>=2.3.3",
        "matplotlib>=3.10.6",
        "shap>=0.48.0",
    ],
)
def explainer_visualization(
    project_files: Input[Model],
    explainer_results: Input[Dataset],
    explainer_plots: Output[Dataset],
    sensitivity: float,
) -> None:
    """Create visualizations for explainer analysis results.

    :param project_files: Input path containing project scripts.
    :param explainer_results: Input path containing explainer analysis output.
    :param explainer_plots: Output path for visualization PNG files.
    :param sensitivity: Sensitivity parameter used by the explainer analysis.
    """
    import subprocess
    from pathlib import Path

    proj_path = Path(project_files.path)
    explainer_results_path = Path(explainer_results.path)
    plots_path = Path(explainer_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "explainer_visualization.py"
    method = "lime" if sensitivity < 0.5 else "shap"
    analysis_file = (
        explainer_results_path / "lime_analysis.json"
        if method == "lime"
        else explainer_results_path / "shap_detailed_results.pickle"
    )

    if not script.exists():
        raise FileNotFoundError(f"Explainer visualization script not found at {script}")
    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Explainer analysis results not found at {analysis_file}"
        )

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_file),
        "--output",
        str(plots_path),
        "--sensitivity",
        str(sensitivity),
    ]
    subprocess.run(cmd, check=True)


@dsl.pipeline(
    name="COPowereD Model Fairness-Bias and Explainer Pipeline",
    description="Runs COPowereD Docker predictions, fairness-bias, and explainer analyses.",
)
def copowered_pipeline(
    github_repo_url: str,
    branch: str = "main",
    sensitivity: float = 0.7,
):
    """Run the COPowereD model fairness/bias and explainer pipeline.

    :param github_repo_url: URL of the GitHub repository containing code and data.
    :param branch: Branch name to pull from.
    :param sensitivity: Sensitivity parameter for explainer analysis.
    """
    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("1000m")
    repo_task.set_cpu_limit("2000m")
    repo_task.set_memory_request("2Gi")
    repo_task.set_memory_limit("4Gi")

    predictions_task = copowered_predictions(data=repo_task.outputs["data"])
    predictions_task.after(repo_task)
    predictions_task.set_caching_options(False)
    predictions_task.set_cpu_request("4000m")
    predictions_task.set_cpu_limit("8000m")
    predictions_task.set_memory_request("6Gi")
    predictions_task.set_memory_limit("10Gi")

    fairness_task = fairness_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        predictions=predictions_task.outputs["predictions"],
    )
    fairness_task.after(predictions_task)
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("2000m")
    fairness_task.set_cpu_limit("4000m")
    fairness_task.set_memory_request("3Gi")
    fairness_task.set_memory_limit("6Gi")

    fairness_viz_task = fairness_visualization(
        project_files=repo_task.outputs["project_files"],
        fairness_results=fairness_task.outputs["fairness_results"],
    )
    fairness_viz_task.after(fairness_task)
    fairness_viz_task.set_caching_options(False)
    fairness_viz_task.set_cpu_request("1000m")
    fairness_viz_task.set_cpu_limit("2000m")
    fairness_viz_task.set_memory_request("2Gi")
    fairness_viz_task.set_memory_limit("4Gi")

    explainer_task = explainer_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        predictions=predictions_task.outputs["predictions"],
        sensitivity=sensitivity,
    )
    explainer_task.after(predictions_task)
    explainer_task.set_caching_options(False)
    explainer_task.set_cpu_request("4000m")
    explainer_task.set_cpu_limit("8000m")
    explainer_task.set_memory_request("6Gi")
    explainer_task.set_memory_limit("10Gi")

    explainer_viz_task = explainer_visualization(
        project_files=repo_task.outputs["project_files"],
        explainer_results=explainer_task.outputs["explainer_results"],
        sensitivity=sensitivity,
    )
    explainer_viz_task.after(explainer_task)
    explainer_viz_task.set_caching_options(False)
    explainer_viz_task.set_cpu_request("1000m")
    explainer_viz_task.set_cpu_limit("2000m")
    explainer_viz_task.set_memory_request("2Gi")
    explainer_viz_task.set_memory_limit("4Gi")


if __name__ == "__main__":
    kfp_compiler = compiler.Compiler()
    kfp_compiler.compile(
        pipeline_func=copowered_pipeline,
        package_path="copowered_pipeline.yaml",
    )
