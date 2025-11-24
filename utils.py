import pandas as pd
import json
from typing import Any, Union, Optional, List, Dict, Literal
from pathlib import Path
import numpy as np
import pickle


def load_csv(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (Union[str, Path]): Path to the CSV file.

    Returns:
        Optional[pd.DataFrame]: Data from the CSV file if successful,
        otherwise None if an error occurs.

    Raises:
        None: Errors are caught and printed instead of raised.
    """

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Could not parse '{file_path}'.")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy data types.
    This encoder extends the standard :class:`json.JSONEncoder` to handle
    NumPy-specific objects that are not JSON serializable by default.

    Supported conversions:
        - np.integer → int
        - np.floating → float
        - np.ndarray → list
        - np.bool_ / bool → bool
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def store_json(data: Any, path: Union[str, Path]) -> None:
    """Store data as a JSON file.

    Args:
        data (Any): The data to be serialized into JSON.
        path (Union[str, Path]): The file path where the JSON will be stored.

    Returns:
        None
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def concat_dfs(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames along columns (axis=1).

    Args:
        df_list (List[pd.DataFrame]): List of DataFrames to concatenate.

    Returns:
        pd.DataFrame: A single DataFrame with columns from all input DataFrames.
    """

    return pd.concat(df_list, axis=1)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file into a Python dictionary.

    Args:
        path (str | Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return data


def is_between(x: float, low: float = 0, high: float = 1) -> bool:
    """
    Check if a value lies within a closed interval [low, high].

    Args:
        x (float): The value to check.
        low (float, optional): Lower bound of the interval. Defaults to 0.
        high (float, optional): Upper bound of the interval. Defaults to 1.

    Returns:
        bool: True if `x` is between `low` and `high`, False otherwise.
    """
    return low <= x <= high


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save any Python object to a pickle file.

    Args:
        data: Any Python object to save
        filepath: Path to save file (str or Path object)
    """

    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load any Python object from a pickle file.

    Args:
        filepath: Path to pickle file (str or Path object)

    Returns:
        The loaded Python object
    """

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data


def select_method(sens: float) -> Literal["shap", "lime"]:
    """
    Select an explainability method based on sensitivity level.
    Values < 0.5 -> "lime".
    Values >= 0.5 -> "shap".

    Args:
        sens (float): Sensitivity parameter in the range [0, 1].

    Returns:
        Literal["shap", "lime"]: The name of the selected
                                explainability method.
    """

    return "lime" if sens < 0.5 else "shap"
