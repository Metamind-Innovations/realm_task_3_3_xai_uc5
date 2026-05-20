import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


class COPowereDWrapper:
    """Wrapper class for the Dockerized COPowereD COPD triage model."""

    MODEL_COLUMNS = [
        "sexe",
        "age",
        "baseline_height",
        "baseline_weight",
        "baseline_bmi",
        "baseline_copd",
        "baseline_heartRate",
        "baseline_spo2",
        "symp_worsening",
        "symp_breath",
        "symp_cough",
        "symp_sputum",
        "heartRate",
        "spo2",
    ]
    CLASS_NAMES = {0: "notNeedMedicalAttention", 1: "needMedicalAttention"}
    DEFAULT_IMAGE = "copd_comunicare:1.0.1"
    MODEL_COMMAND = (
        "java -cp /app/sparkServer-assembly-1.2.0.jar "
        "MLProjects.bpco.triage.models.CopdComunicare_001 "
        "--input_folder /app/data/data "
        "--output_folder /app/data/result"
    )

    def __init__(
        self,
        image: Optional[str] = None,
        threshold: float = 0.5,
        feature_names: Optional[list] = None,
        one_dim_preds: bool = False,
        docker_executable: str = "docker",
        in_docker: bool = False,
    ) -> None:
        """Initialize the Dockerized COPowereD model wrapper.

        :param image: Docker image name. Defaults to the
            ``COPOWERED_MODEL_IMAGE`` environment variable or the local
            ``copd_comunicare:1.0.1`` image loaded with ``docker load``.
        :param threshold: Classification threshold.
        :param feature_names: Feature names used when explainers provide NumPy
            arrays instead of pandas DataFrames.
        :param one_dim_preds: Return only positive-class probabilities when
            ``True``.
        :param docker_executable: Docker command executable.
        :param in_docker: When ``True``, the wrapper is running inside the
            COPowereD container and invokes the Java command directly instead
            of launching a ``docker run`` subprocess.
        """
        self.image = image or os.getenv("COPOWERED_MODEL_IMAGE", self.DEFAULT_IMAGE)
        self.threshold = threshold
        self.feature_names = feature_names
        self.one_dim_preds = one_dim_preds
        self.docker_executable = docker_executable
        self.in_docker = in_docker
        self._last_probabilities = None
        self._last_predictions = None
        self._last_response = None

    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict class probabilities for input tabular data.

        The Docker image expects ``/app/data/data/data.csv`` and writes
        ``/app/data/result/result.csv`` with a single ``proba`` column.

        :param data: DataFrame or array with model input columns.
        :return: Probability array with shape ``(n_samples, 2)`` unless
            ``one_dim_preds`` is ``True``.
        :raises ValueError: If the input data or model output is inconsistent.
        :raises RuntimeError: If Docker execution fails.
        """

        if isinstance(data, np.ndarray):
            if not self.feature_names:
                raise ValueError(
                    "Input data is np.ndarray. Feature names should be passed."
                )
            data = pd.DataFrame(data=data, columns=self.feature_names)
        else:
            data = data.copy()

        missing_columns = [
            column for column in self.MODEL_COLUMNS if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                "Input data is missing columns required by the Dockerized model: "
                f"{', '.join(missing_columns)}"
            )

        model_data = data.loc[:, self.MODEL_COLUMNS]

        with tempfile.TemporaryDirectory(prefix="copowered_model_") as tmp_dir:
            data_dir = Path(tmp_dir).resolve()
            input_dir = data_dir / "data"
            output_dir = data_dir / "result"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_csv = input_dir / "data.csv"
            result_csv = output_dir / "result.csv"
            model_data.to_csv(input_csv, index=False)

            if self.in_docker:
                # Already inside the container — call the Java model directly.
                shell_cmd = (
                    "cd /app && "
                    "printf '%s\\n' "
                    "'log4j.rootLogger=ERROR, stdout' "
                    "'log4j.appender.stdout=org.apache.log4j.ConsoleAppender' "
                    "'log4j.appender.stdout.Target=System.out' "
                    "'log4j.appender.stdout.layout=org.apache.log4j.PatternLayout' "
                    "> log4j.properties && "
                    "java -cp /app/sparkServer-assembly-1.2.0.jar "
                    "MLProjects.bpco.triage.models.CopdComunicare_001 "
                    f"--input_folder {input_dir.as_posix()} "
                    f"--output_folder {output_dir.as_posix()}"
                )
                completed = subprocess.run(
                    shell_cmd,
                    shell=True,
                    capture_output=True,
                    check=False,
                    text=True,
                )
            else:
                command = [
                    self.docker_executable,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "sh",
                    "-v",
                    f"{data_dir.as_posix()}:/app/data",
                    self.image,
                    "-c",
                    (
                        "cd /app && "
                        "printf '%s\\n' "
                        "'log4j.rootLogger=ERROR, stdout' "
                        "'log4j.appender.stdout=org.apache.log4j.ConsoleAppender' "
                        "'log4j.appender.stdout.Target=System.out' "
                        "'log4j.appender.stdout.layout=org.apache.log4j.PatternLayout' "
                        "> log4j.properties && "
                        f"{self.MODEL_COMMAND}"
                    ),
                ]
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    text=True,
                )

            if completed.returncode == 0 and result_csv.exists():
                result_df = pd.read_csv(result_csv)
                if "proba" not in result_df.columns:
                    raise ValueError(
                        "Dockerized model output must contain a 'proba' column."
                    )
                if len(result_df) != len(model_data):
                    raise ValueError(
                        "Dockerized model output row count does not match "
                        "the input row count."
                    )

                probabilities = result_df["proba"].to_numpy(dtype=float)
                self._last_response = result_df.copy()
            else:
                model_error = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "Dockerized model did not create result.csv."
                )

                raise RuntimeError(
                    "Dockerized model failed: "
                    f"{model_error}"
                )

        all_probabilities = np.column_stack(
            [1.0 - probabilities, probabilities]
        )

        if self.one_dim_preds:
            self._last_probabilities = probabilities
            return probabilities

        self._last_probabilities = all_probabilities
        return all_probabilities

    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Predict binary class labels for input tabular data.

        :param data: DataFrame or array with model input columns.
        :return: Array of binary class labels.
        """
        probabilities = self.predict_proba(
            data=data,
        )
        positive_probabilities = (
            probabilities if probabilities.ndim == 1 else probabilities[:, 1]
        )
        predictions = (positive_probabilities > self.threshold).astype(int)
        self._last_predictions = predictions

        return predictions

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame to the Docker model input column order.

        :param df: DataFrame with model input columns.
        :return: DataFrame with columns ordered for the Dockerized model.
        """
        missing_columns = [
            column for column in self.MODEL_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                "Input data is missing columns required by the Dockerized model: "
                f"{', '.join(missing_columns)}"
            )

        return df.loc[:, self.MODEL_COLUMNS].copy()

    def get_last_response(self) -> Optional[pd.DataFrame]:
        """Get the raw ``result.csv`` DataFrame from the last prediction.

        :return: Last model output DataFrame, or ``None`` if no predictions
            have been made.
        """
        if self._last_response is None:
            return None

        return self._last_response.copy()

    def get_class_names(self) -> Dict[int, str]:
        """Get mapping of class indices to class names.

        :return: Dictionary mapping class indices to descriptive names.
        """
        return self.CLASS_NAMES.copy()
