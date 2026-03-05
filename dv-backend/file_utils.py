"""
Utility functions for file handling (CSV, Parquet, etc.)
"""
from pathlib import Path
from typing import Union

import pandas as pd


def read_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read a dataframe from CSV or Parquet file.

    Args:
        file_path: Path to the data file (.csv or .parquet)

    Returns:
        pandas DataFrame

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file format from extension
    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(file_path)
    elif suffix in ['.parquet', '.pq']:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .parquet, .pq"
        )


def get_data_files(directory: Path, extensions: list = None) -> list:
    """
    Get all data files (CSV, Parquet) in a directory.

    Args:
        directory: Directory to search
        extensions: List of extensions to search for (default: ['.csv', '.parquet', '.pq'])

    Returns:
        List of Path objects
    """
    if extensions is None:
        extensions = ['.csv', '.parquet', '.pq']

    data_files = []
    for ext in extensions:
        data_files.extend(directory.glob(f"*{ext}"))

    return sorted(data_files)


def get_file_format(file_path: Union[str, Path]) -> str:
    """
    Get the format of a data file.

    Args:
        file_path: Path to the file

    Returns:
        'csv', 'parquet', or 'unknown'
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == '.csv':
        return 'csv'
    elif suffix in ['.parquet', '.pq']:
        return 'parquet'
    else:
        return 'unknown'
