import pandas as pd
import json
from typing import Any, Union, Optional, List
from pathlib import Path
import numpy as np


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
