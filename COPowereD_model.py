from typing import Dict, Optional, Any, Union
import requests
import pandas as pd
import numpy as np
import time


class COPowereDWrapper:
    """
    Wrapper class for the COPowereD COPD triage prediction API.
    """

    # Mapping from DataFrame columns to API JSON format
    COLUMN_MAPPING = {
        "Gender": {
            "code": "46098-0",
            "display": "sex",
            "system": "http://loinc.org",
            "instant": None,
        },
        "Age": {
            "code": "63900-5",
            "display": "Current age or age at death",
            "system": "http://loinc.org",
            "instant": None,
        },
        "Height": {
            "code": "8302-2",
            "display": "Body height",
            "system": "http://loinc.org",
            "instant": "baseline",
        },
        "Weight": {
            "code": "3141-9",
            "display": "Body Weight",
            "system": "http://loinc.org",
            "instant": "baseline",
        },
        "BMI": {
            "code": "39156-5",
            "display": "Body mass index",
            "system": "http://loinc.org",
            "instant": "baseline",
        },
        "b_COPD": {
            "code": "13645005",
            "display": "COPD Gold stage",
            "system": "http://snomed.info/sct",
            "instant": "baseline",
        },
        "b_HeartRate": {
            "code": "8867-4",
            "display": "Heart rate baseline",
            "system": "http://loinc.org",
            "instant": "baseline",
        },
        "b_SPO2": {
            "code": "20564-1",
            "display": "Oxygen saturation baseline",
            "system": "http://loinc.org",
            "instant": "baseline",
        },
        "s_Worsening": {
            "code": "275723000",
            "display": "Deteriorating condition",
            "system": "http://snomed.info/sct",
            "instant": None,
        },
        "s_Breath": {
            "code": "267036007",
            "display": "Dyspnea",
            "system": "http://snomed.info/sct",
            "instant": None,
        },
        "s_Cough": {
            "code": "49727002",
            "display": "Cough",
            "system": "http://snomed.info/sct",
            "instant": None,
        },
        "s_Sputum": {
            "code": "248595008_365445003",
            "display": "Sputum : Volume and Color",
            "system": "http://snomed.info/sct",
            "instant": None,
        },
        "c_HeartRate": {
            "code": "8867-4",
            "display": "Heart rate",
            "system": "http://loinc.org",
            "instant": None,
        },
        "c_SPO2": {
            "code": "20564-1",
            "display": "Oxygen saturation",
            "system": "http://loinc.org",
            "instant": None,
        },
    }

    # Class names
    CLASS_NAMES = {0: "notNeedMedicalAttention", 1: "needMedicalAttention"}

    def __init__(
        self,
        url: str = "https://canalytics.comunicare.io/api/prediction",
        access_token: Optional[str] = None,
        threshold: float = 0.5,
        project: str = "BpcoTriagingBinary",
        algo: str = "MLGBTPipeline",
        version: str = "0.0.1",
        feature_names: list = None,
        one_dim_preds: bool = False,
    ):
        """
        Initialize the COPowereD API wrapper.

        Args:
            url (str): API endpoint URL
            access_token (Optional[str]): Optional authentication token.
            threshold (float): Classification threshold (default: 0.5).
            project (str): Model project name (default: BpcoTriagingBinary).
            algo (str): Algorithm name (default: MLGBTPipeline).
            version (str): Model version (default: 0.0.1).
            feature_names (list): List with names of features (default: None).
            one_dim_preds (bool): Flag to return probs only for positive class (default: False).
        """

        self.url = url
        self.threshold = threshold
        self.project = project
        self.algo = algo
        self.version = version
        self.feature_names = feature_names
        self.one_dim_preds = one_dim_preds

        self.headers = {"Content-Type": "application/json"}
        if access_token:
            self.headers["x-access-token"] = access_token

        # Store last predictions
        self._last_probabilities = None
        self._last_predictions = None
        self._last_response = None

    def _dataframe_to_json(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform pandas DataFrame to API JSON format.
        Converts patient data from tabular format to the nested JSON structure
        required by the COPowereD API.

        Args:
            df (pd.DataFrame): DataFrame with patient data. An 'id' column
                               will be added if not present.

        Returns:
            Dict[str, Any]: Dictionary in API request format.
        """

        # Add patient IDs if not present
        if "id" not in df.columns:
            df = df.copy()
            df["id"] = range(len(df))

        # Get only columns that exist in mapping
        valid_cols = [col for col in df.columns if col in self.COLUMN_MAPPING]

        # Convert to list of dictionaries (faster than iterrows)
        records = df.to_dict("records")

        # Build observations using list comprehension
        observations = [
            {
                "subject": {"reference": f"patient_{row['id']}"},
                "component": [
                    {
                        "valueQuantity": {
                            "value": (
                                float(row[col])
                                if not isinstance(row[col], (int, float))
                                else row[col]
                            )
                        },
                        "code": {
                            "coding": [
                                {
                                    "code": self.COLUMN_MAPPING[col]["code"],
                                    "display": self.COLUMN_MAPPING[col]["display"],
                                    "system": self.COLUMN_MAPPING[col]["system"],
                                }
                            ]
                        },
                        **(
                            {"instant": self.COLUMN_MAPPING[col]["instant"]}
                            if self.COLUMN_MAPPING[col]["instant"] is not None
                            else {}
                        ),
                    }
                    for col in valid_cols
                    if not pd.isna(row[col])
                ],
            }
            for row in records
        ]

        payload = {
            "methods": [
                {"project": self.project, "algo": self.algo, "version": self.version}
            ],
            "observations": observations,
        }

        return payload

    def _extract_probabilities(self, response_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract probability matrix from API response.
        Parses the nested API response structure and extracts class probabilities
        for each patient, filtering for GBTP (Gradient Boosted Tree Pipeline)
        predictions.

        Args:
            response_data (Dict[str, Any]): API response data containing predictions
                                            for one or more patients.

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with class probabilities.
                - Column 0: probability of notNeedMedicalAttention
                - Column 1: probability of needMedicalAttention
        """

        probabilities = [
            [
                # Build prob_dict using dict comprehension
                prob_dict.get("notNeedMedicalAttention", 0.0),
                prob_dict.get("needMedicalAttention", 0.0),
            ]
            for patient_data in response_data["data"]
            for prob_dict in [
                {
                    pred["outcome"]["coding"][0]["code"]: pred["probabilityDecimal"]
                    for pred in patient_data["prediction"]
                    if pred.get("rationale") == "GBTP"
                }
            ]
        ]

        return np.array(probabilities)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        num_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 1000,
    ) -> np.ndarray:
        """
        Predict class probabilities for input DataFrame with retry mechanism.
        Makes API request to get probability predictions for each patient.
        Automatically retries on network/request failures with exponential backoff.
        Processes large datasets in chunks to avoid API limits.

        Args:
            data (Union[pd.DataFrame, np.ndarray): DataFrame or array with patient
                            data containing columns mapped in COLUMN_MAPPING.
            num_retries (int): Number of retry attempts on failure (default: 3).
            retry_delay (float): Initial delay between retries in seconds.
            batch_size (int): Maximum number of samples per API request (default: 1000).

        Returns:
            np.ndarray: Array of shape (n_samples, 2) with class probabilities.
                - Column 0: probability of notNeedMedicalAttention (class 0)
                - Column 1: probability of needMedicalAttention (class 1)
                If one_dim_preds is True, returns shape (n_samples,) with only
                positive class probabilities.
        """

        # Check input type
        if isinstance(data, np.ndarray):
            if not self.feature_names:
                raise ValueError(
                    "Input data is np.ndarray. Feature names should be passed."
                )
            data = pd.DataFrame(data=data, columns=self.feature_names)

        # Total number of samples and batches
        n_samples = len(data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        all_probabilities = []

        # Predictions in batches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            batch_data = data.iloc[start_idx:end_idx]

            # Transform DataFrame to JSON
            payload = self._dataframe_to_json(batch_data)

            # Make API request with retries
            for attempt in range(num_retries):
                try:
                    response = requests.post(
                        self.url, headers=self.headers, json=payload
                    )
                    response.raise_for_status()

                    result = response.json()

                    if not result["success"]:
                        raise ValueError(
                            f"API returned error: {result.get('message', 'Unknown error')}"
                        )

                    # Store response
                    self._last_response = result

                    # Extract probabilities
                    batch_probabilities = self._extract_probabilities(result)
                    all_probabilities.append(batch_probabilities)

                    break

                except requests.exceptions.RequestException as e:
                    if attempt < num_retries - 1:
                        wait_time = retry_delay * (2**attempt)  # Exponential backoff
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(
                            f"API request failed for batch {batch_idx + 1}/{n_batches} "
                            f"(samples {start_idx}-{end_idx}) after {num_retries} attempts: {e}"
                        )

        # Merge all batch results maintaining order
        all_probabilities = np.vstack(all_probabilities)

        if self.one_dim_preds:
            all_probabilities = all_probabilities[:, 1]

        self._last_probabilities = all_probabilities

        return all_probabilities

    def predict(
        self, df: pd.DataFrame, num_retries: int = 3, retry_delay: float = 1.0
    ) -> np.ndarray:
        """
        Predict class labels for input DataFrame.
        Converts probability predictions to binary class labels using the
        configured threshold.

        Args:
            df (pd.DataFrame): DataFrame with patient data containing columns
                mapped in COLUMN_MAPPING (Gender, Age, Height, Weight, etc.).
            num_retries (int): Number of retry attempts on failure (default: 3).
            retry_delay (float): Initial delay between retries in seconds (default: 1.0).

        Returns:
            np.ndarray: Array of shape (n_samples,) with class labels:
                - 0: notNeedMedicalAttention
                - 1: needMedicalAttention
        """

        probabilities = self.predict_proba(
            df=df, num_retries=num_retries, retry_delay=retry_delay
        )

        # Apply threshold to get predictions
        predictions = (probabilities[:, 1] > self.threshold).astype(int)
        self._last_predictions = predictions

        return predictions

    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform DataFrame to API JSON format without making predictions.

        Args:
            df (pd.DataFrame): DataFrame with patient data

        Returns:
            Dict[str, Any]: Dictionary in API request format with complete
                nested structure ready for API submission
        """

        return self._dataframe_to_json(df)

    def get_last_response(self) -> Optional[Dict[str, Any]]:
        """
        Get the raw API response from the last prediction.

        Returns:
            Optional[Dict[str, Any]]: Complete API response dictionary including
                predictions, probabilities, etc. Returns None if no
                predictions have been made yet.
        """

        return self._last_response

    def get_class_names(self) -> Dict[int, str]:
        """
        Get mapping of class indices to class names.

        Returns:
            Dict[int, str]: Dictionary mapping class indices to descriptive names.
        """

        return self.CLASS_NAMES.copy()
