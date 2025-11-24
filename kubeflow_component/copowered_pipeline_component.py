from kfp import dsl, compiler
from kfp.dsl import Input, Output, Dataset, Model


# -----------------------
# Step 1: Download Repo
# -----------------------
@dsl.component(base_image="python:3.13-slim")
def download_repo(
    github_repo_url: str,
    project_files: Output[Model],
    data: Output[Dataset],
    branch: str = "main",
) -> None:
    """Download specific scripts and data from a GitHub repository.
    This component clones a GitHub repository, copies selected Python scripts
    into the `project_files` output, and the `data` folder into the `data` output.

    Args:
        github_repo_url (str): URL of the GitHub repository to clone.
        project_files (Output[Model]): Output path for project scripts.
        data (Output[Dataset]): Output path for data folder.
        branch (str): Branch name to pull from (defaults to 'main').
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

    print(f"Cloning repo {github_repo_url} (branch: {branch})...")
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

    # List of files to copy
    files_to_copy = [
        "COPowereD_model.py",
        "explainer.py",
        "fairness_bias_analysis.py",
        "utils.py",
    ]

    # Copy specific scripts
    proj_path = Path(project_files.path)
    proj_path.mkdir(parents=True, exist_ok=True)
    for filename in files_to_copy:
        src_file = repo_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, proj_path / src_file.name)
            print(f"Copied {filename} to project_files")
        else:
            print(f"Warning: {filename} not found in repo")

    # Copy data folder
    data_path = Path(data.path)
    data_path.mkdir(parents=True, exist_ok=True)
    src_data_path = repo_dir / "data"
    if (src_data_path).exists():
        shutil.copytree(src_data_path, data_path, dirs_exist_ok=True)
        print("Copied data folder")


# -----------------------
# Step 2: Fairness Analysis
# -----------------------
@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["numpy==2.3.3", "pandas==2.3.3", "scikit-learn==1.7.2"],
)
def fairness_analysis(
    project_files: Input[Model],
    data: Input[Dataset],
    fairness_results: Output[Dataset],
) -> None:
    """Run fairness and bias analysis for the COPowereD model.
    This component installs required Python packages, executes the
    `fairness_bias_analysis.py` script from the project repository, and writes
    results to the `fairness_results` output.

    Args:
        project_files (Input[Model]): Input path containing project scripts.
        data (Input[Dataset]): Input path containing `tabular.csv` and `pred.csv`.
        fairness_results (Output[Dataset]): Output path for fairness analysis results (JSON).
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(fairness_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "fairness_bias_analysis.py"
    if not script.exists():
        raise FileNotFoundError(f"Fairness analyzer script not found at {script}")

    print(f"Running fairness analysis with {script}")

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(data_path / "tabular_data.csv"),
        "--pred_target",
        str(data_path / "pred.csv"),
        "--output",
        str(results_path / "fairness_analysis.json"),
    ]
    subprocess.run(cmd, check=True)

    print(f"Fairness analysis finished. Results saved to {results_path}")


# -----------------------
# Step 3: Fairness Bias Visualization
# -----------------------
@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=["numpy==2.3.3", "pandas==2.3.3", "matplotlib==3.10.6"],
)
def fairness_visualization(
    project_files: Input[Model],
    fairness_results: Input[Dataset],
    fairness_plots: Output[Dataset],
) -> None:
    """Create visualizations for fairness and bias analysis results.
    This component generates bar charts showing fairness metrics across
    demographic groups.

    Args:
        project_files (Input[Model]): Input path containing project scripts.
        fairness_results (Input[Dataset]): Input path containing fairness_analysis.json.
        fairness_plots (Output[Dataset]): Output path for visualization PNG files.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    results_path = Path(fairness_results.path)
    plots_path = Path(fairness_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
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


# -----------------------
# Step 4: Explainer Analysis
# -----------------------
@dsl.component(
    base_image="python:3.13-slim",
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
    This component installs the required Python packages, executes
    the `explainer.py` script from the project repository,
    and writes the results to the `explainer_results` output.

    Args:
        project_files (Input[Model]): Input path containing project scripts.
        data (Input[Dataset]): Input path containing `tabular_data.csv`.
        explainer_results (Output[Dataset]): Output path for explainer results.
        sensitivity (float): Sensitivity parameter for the explainer script.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    data_path = Path(data.path)
    results_path = Path(explainer_results.path)
    results_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
    script = proj_path / "explainer.py"
    if not script.exists():
        raise FileNotFoundError(f"Explainer script not found at {script}")

    print(f"Running explainer analysis with {script}")

    cmd = [
        "python",
        str(script),
        "--tabular_data",
        str(data_path / "tabular_data.csv"),
        "--sensitivity",
        str(sensitivity),
        "--output",
        str(results_path),
    ]
    subprocess.run(cmd, check=True)

    print(f"Explainer analysis finished. Results saved to {results_path}")


# -----------------------
# Step 5: Explainer Visualization
# -----------------------
@dsl.component(
    base_image="python:3.13-slim",
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
    This component generates plots (SHAP beeswarm/bar or LIME bar charts)
    based on the explainability method used.

    Args:
        project_files (Input[Model]): Input path containing project scripts.
        explainer_results (Input[Dataset]): Input path containing explainer results
            (shap_detailed_results.pickle or lime_analysis.json).
        explainer_plots (Output[Dataset]): Output path for visualization PNG files.
        sensitivity (float): Sensitivity parameter to determine which method was used.
    """
    from pathlib import Path
    import subprocess

    # Prepare paths
    proj_path = Path(project_files.path)
    explainer_results_path = Path(explainer_results.path)
    plots_path = Path(explainer_plots.path)
    plots_path.mkdir(parents=True, exist_ok=True)

    # Prepare script and arguments
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
# -----------------------
# Define Pipeline
# -----------------------
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

    Args:
        github_repo_url (str): URL of the GitHub repository containing the COPowereD code and data.
        branch (str): Branch name to pull from (defaults to 'main').
        sensitivity (float): Sensitivity parameter for the explainer analysis. Defaults to 0.7.
    """

    # Step 1: Download repository
    repo_task = download_repo(github_repo_url=github_repo_url, branch=branch)
    repo_task.set_caching_options(False)
    repo_task.set_cpu_request("1000m")
    repo_task.set_cpu_limit("2000m")
    repo_task.set_memory_request("2Gi")
    repo_task.set_memory_limit("4Gi")

    # Step 2: Fairness analysis
    fairness_task = fairness_analysis(
        project_files=repo_task.outputs["project_files"],
        data=repo_task.outputs["data"],
    )
    fairness_task.after(repo_task)
    fairness_task.set_caching_options(False)
    fairness_task.set_cpu_request("1000m")
    fairness_task.set_cpu_limit("2000m")
    fairness_task.set_memory_request("2Gi")
    fairness_task.set_memory_limit("4Gi")

    # Step 3: Fairness visualization
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

    # Step 4: Explainer analysis
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

    # Step 5: Explainer visualization
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
    compiler = compiler.Compiler()
    compiler.compile(
        pipeline_func=copowered_pipeline, package_path="copowered_pipeline.yaml"
    )
