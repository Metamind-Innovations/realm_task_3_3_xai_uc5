from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model

# Insert your dockerhub image below (e.g. "docker.io/<username>/copd_comunicare:1.0.0")
DOCKER_IMAGE = "<docker_image>"

# Filename of the validation dataset in the data/ folder of the repo
TABULAR_DATA_FILENAME = "validation_dataset_REALM_20250716.csv"


@dsl.component(base_image="python:3.14-slim")
def download_repo(
        github_repo_url: str,
        project_files: Output[Model],
        data: Output[Dataset],
        branch: str = "main",
) -> None:
    """Download specific scripts and data from a GitHub repository.

    :param github_repo_url: URL of the GitHub repository to clone.
    :param project_files: Output path for project scripts.
    :param data: Output path for data folder.
    :param branch: Branch name to pull from (defaults to 'main').
    """
    import shutil
    from pathlib import Path
    import subprocess

    repo_dir = Path("/tmp/repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("Installing git...")
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

    # Copy everything from src/ folder to project_files
    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)
    src_folder = repo_dir / "src"

    if src_folder.exists():
        for item in src_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, proj_path / item.name)
                print(f"Copied src/{item.name}")
            elif item.is_dir():
                shutil.copytree(item, proj_path / item.name, dirs_exist_ok=True)
                print(f"Copied src/{item.name}/ directory")
    else:
        print("Warning: src/ folder not found in repo")

    # Verify all required project files exist
    required_files = [
        "COPowereD_model.py",
        "explainer.py",
        "fairness_bias_analysis.py",
        "utils.py",
        "fairness_bias_visualization.py",
        "explainer_visualization.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = proj_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"ERROR: Required file missing: {file_path}")
        else:
            print(f"✓ Verified: {file_path}")

    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    # Copy everything inside data/ folder (data.csv + validation_dataset_REALM_20250716.csv)
    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"

    if src_data_path.exists():
        for item in src_data_path.iterdir():
            if item.is_file():
                shutil.copy2(item, data_path / item.name)
                print(f"Copied data/{item.name}")
            elif item.is_dir():
                shutil.copytree(item, data_path / item.name, dirs_exist_ok=True)
                print(f"Copied data/{item.name}/ directory")
    else:
        print("Warning: data/ folder not found in repo")

    # Verify required data files
    required_data_files = ["data.csv", "validation_dataset_REALM_20250716.csv"]
    for filename in required_data_files:
        if (data_path / filename).exists():
            print(f"✓ Verified: data/{filename}")
        else:
            print(f"WARNING: data/{filename} not found")


@dsl.container_component
def copowered_predictions(
        data: Input[Dataset],
        predictions: Output[Dataset],
):
    """Run COPowereD model predictions on patient data using the Docker image.

    Reads data/data.csv, writes result.csv with a single 'proba' column
    containing the predicted probability of needing medical attention.

    :param data: Input dataset path containing data.csv (Docker-formatted input).
    :param predictions: Output path for result.csv with proba column.
    """
    command_str = f"""
        set -e
        mkdir -p /app/data
        mkdir -p {predictions.path}
        cp {data.path}/data.csv /app/data/data.csv
        cd /app && python predict.py
        cp /app/data/result.csv {predictions.path}/result.csv
        [ -f "{predictions.path}/result.csv" ] || exit 1
    """

    return dsl.ContainerSpec(
        image=DOCKER_IMAGE,
        command=["sh", "-c"],
        args=[command_str]
    )


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=["numpy==2.3.3", "pandas==2.3.3", "scikit-learn==1.7.2"],
)
def fairness_analysis(
        project_files: Input[Model],
        data: Input[Dataset],
        predictions: Input[Dataset],
        fairness_results: Output[Dataset],
) -> None:
    """Run fairness and bias analysis for the COPowereD model.

    Converts probability predictions from result.csv to binary labels
    (threshold 0.5) before running the fairness analysis script.

    :param project_files: Input path containing project scripts.
    :param data: Input path containing validation_dataset_REALM_20250716.csv
        with ground truth labels and demographic columns.
    :param predictions: Input path containing result.csv with proba column
        (output of the COPowereD Docker model).
    :param fairness_results: Output path for fairness analysis results (JSON).
    """
    from pathlib import Path
    import subprocess
    import pandas as pd

    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    predictions_path = Path(predictions.path)
    results_path = Path(fairness_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "fairness_bias_analysis.py"
    if not script.exists():
        raise FileNotFoundError(f"Fairness analyzer script not found at {script}")

    # Convert probability predictions to binary labels (threshold = 0.5)
    result_csv = predictions_path / "result.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Predictions file not found: {result_csv}")

    pred_df = pd.read_csv(str(result_csv))
    pred_df["label"] = (pred_df["proba"] >= 0.5).astype(int)
    pred_csv = results_path / "pred.csv"
    pred_df[["label"]].to_csv(str(pred_csv), index=False)
    print(f"Converted {len(pred_df)} probability predictions to binary labels.")

    tabular_csv = data_path / "validation_dataset_REALM_20250716.csv"
    if not tabular_csv.exists():
        raise FileNotFoundError(f"Tabular data file not found: {tabular_csv}")

    print(f"Running fairness analysis with {script}")

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

    print(f"Fairness analysis finished. Results saved to {results_path}")


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=["numpy==2.3.3", "pandas==2.3.3", "matplotlib==3.10.6"],
)
def fairness_visualization(
        project_files: Input[Model],
        fairness_results: Input[Dataset],
        fairness_plots: Output[Dataset],
) -> None:
    """Create visualizations for fairness and bias analysis results.

    :param project_files: Input path containing project scripts.
    :param fairness_results: Input path containing fairness_analysis.json.
    :param fairness_plots: Output path for visualization PNG files.
    """
    from pathlib import Path
    import subprocess

    proj_path = Path(project_files.path)
    results_path = Path(fairness_results.path)
    plots_path = Path(fairness_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "fairness_bias_visualization.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Fairness Bias visualization script not found at {script}"
        )

    analysis_results_file = results_path / "fairness_analysis.json"
    if not analysis_results_file.exists():
        raise FileNotFoundError(
            f"Fairness Bias analysis results not found at {analysis_results_file}"
        )

    print(f"Running fairness bias visualization with {script}")

    cmd = [
        "python",
        str(script),
        "--analysis_results",
        str(analysis_results_file),
        "--output",
        str(plots_path),
    ]
    subprocess.run(cmd, check=True)

    print(f"Fairness Bias visualization completed. Plots saved to {plots_path}")


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=[
        "numpy==2.3.3",
        "pandas==2.3.3",
        "scikit-learn==1.7.2",
        "requests==2.32.5",
        "shap==0.48.0",
        "lime==0.2.0.1",
        "tqdm==4.67.1",
    ],
)
def explainer_analysis(
        project_files: Input[Model],
        data: Input[Dataset],
        explainer_results: Output[Dataset],
        sensitivity: float,
) -> None:
    """Run explainer analysis on the COPowereD model.

    :param project_files: Input path containing project scripts.
    :param data: Input path containing validation_dataset_REALM_20250716.csv.
    :param explainer_results: Output path for explainer results.
    :param sensitivity: Sensitivity parameter for the explainer script.
    """
    from pathlib import Path
    import subprocess

    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(explainer_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "explainer.py"
    if not script.exists():
        raise FileNotFoundError(f"Explainer script not found at {script}")

    tabular_csv = data_path / "validation_dataset_REALM_20250716.csv"
    if not tabular_csv.exists():
        raise FileNotFoundError(f"Tabular data file not found: {tabular_csv}")

    print(f"Running explainer analysis with {script}")

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(tabular_csv),
        "--sensitivity",
        str(sensitivity),
        "--output",
        str(results_path),
    ]
    subprocess.run(cmd, check=True)

    print(f"Explainer analysis finished. Results saved to {results_path}")


@dsl.component(
    base_image="python:3.14-slim",
    packages_to_install=[
        "numpy==2.3.3",
        "pandas==2.3.3",
        "matplotlib==3.10.6",
        "shap==0.48.0",
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
    :param explainer_results: Input path containing explainer results
        (shap_detailed_results.pickle or lime_analysis.json).
    :param explainer_plots: Output path for visualization PNG files.
    :param sensitivity: Sensitivity parameter to determine which method was used.
    """
    from pathlib import Path
    import subprocess

    proj_path = Path(project_files.path)
    explainer_results_path = Path(explainer_results.path)
    plots_path = Path(explainer_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    script = proj_path / "explainer_visualization.py"
    if not script.exists():
        raise FileNotFoundError(f"Explainer visualization script not found at {script}")

    # Determine which analysis file to use based on sensitivity
    if sensitivity < 0.5:
        # lime
        analysis_file = explainer_results_path / "lime_analysis.json"
        method = "lime"
    elif sensitivity >= 0.5:
        # shap
        analysis_file = explainer_results_path / "shap_detailed_results.pickle"
        method = "shap"

    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Explainer analysis results not found at {analysis_file}"
        )

    print(f"Running explainer visualization with {script} (method: {method})")

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

    print(f"Explainer visualization completed. Plots saved to {plots_path}")


# -----------------------
# Define Pipeline
# -----------------------
@dsl.pipeline(
    name="COPowereD Model Fairness-Bias and Explainer Pipeline",
    description="Runs fairness-bias and explainer analyses.",
)
def copowered_pipeline(
        github_repo_url: str,
        branch: str = "main",
        sensitivity: float = 0.7,
):
    """Pipeline to run COPowereD model fairness/bias and explainer analyses.

    :param github_repo_url: URL of the GitHub repository containing the COPowereD code and data.
    :param branch: Branch name to pull from (defaults to 'main').
    :param sensitivity: Sensitivity parameter for the explainer analysis. Defaults to 0.7.
    """
    # Step 1: Download repository
    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("1000m")
    repo_task.set_cpu_limit("2000m")
    repo_task.set_memory_request("2Gi")
    repo_task.set_memory_limit("4Gi")

    # Step 2: Run COPowereD model predictions (result.csv with proba column)
    predictions_task = copowered_predictions(
        data=repo_task.outputs["data"]
    )
    predictions_task.after(repo_task)
    predictions_task.set_caching_options(False)
    predictions_task.set_cpu_request("2000m")
    predictions_task.set_cpu_limit("4000m")
    predictions_task.set_memory_request("4Gi")
    predictions_task.set_memory_limit("8Gi")

    # Step 3: Fairness analysis (converts proba → binary labels internally)
    fairness_task = fairness_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        predictions=predictions_task.outputs["predictions"],
    )
    fairness_task.after(predictions_task)
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("1000m")
    fairness_task.set_cpu_limit("2000m")
    fairness_task.set_memory_request("2Gi")
    fairness_task.set_memory_limit("4Gi")

    # Step 4: Fairness visualization
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

    # Step 5: Explainer analysis (runs in parallel with fairness after Step 2)
    explainer_task = explainer_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
        sensitivity=sensitivity,
    )
    explainer_task.after(repo_task)
    explainer_task.set_caching_options(False)
    explainer_task.set_cpu_request("1000m")
    explainer_task.set_cpu_limit("2000m")
    explainer_task.set_memory_request("2Gi")
    explainer_task.set_memory_limit("4Gi")

    # Step 6: Explainer visualization
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
    kfp_compiler.compile(pipeline_func=copowered_pipeline, package_path="copowered_pipeline.yaml")
