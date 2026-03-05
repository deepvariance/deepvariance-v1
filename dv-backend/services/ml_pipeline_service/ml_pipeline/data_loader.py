"""
Data loading functionality for the ML pipeline.
"""

import os
import pandas as pd
import requests
from . import ssl_bypass


class CSVLoader:
    """
    A class to handle loading CSV data from local and remote sources.
    """

    @staticmethod
    def load_local(filepath: str) -> pd.DataFrame:
        """
        Load data from a local CSV file.

        Args:
            filepath (str): Path to the local CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                raise ValueError("The loaded dataset is empty")
            return df
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")

    @staticmethod
    def load_remote(url: str) -> pd.DataFrame:
        """
        Load data from a remote CSV file.

        Args:
            url (str): URL to the remote CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            ValueError: If URL is invalid or data format is incorrect
            requests.RequestException: If network error occurs
        """
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Create a temporary file to store the content
            with open("temp.csv", "wb") as f:
                f.write(response.content)

            # Load the DataFrame
            df = pd.read_csv("temp.csv")

            # Clean up
            os.remove("temp.csv")

            if df.empty:
                raise ValueError("The downloaded dataset is empty")

            return df

        except requests.RequestException as e:
            raise ValueError(f"Error downloading CSV: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing CSV: {str(e)}")

    @staticmethod
    def save_dataset(df: pd.DataFrame, filepath: str) -> None:
        """
        Save DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (str): Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)


class DemoDataLoader:
    """
    Loader for demo datasets from UCI ML Repository.
    """

    @staticmethod
    def load_diabetes_dataset() -> pd.DataFrame:
        """
        Load the CDC Diabetes Health Indicators dataset from UCI ML Repository.

        Returns:
            pd.DataFrame: Combined features and target in a single DataFrame

        Raises:
            ValueError: If there's an error fetching or processing the data
        """
        try:
            # Fetch dataset (ID 891 is the CDC Diabetes Health Indicators)
            dataset = ssl_bypass.fetch_with_ssl_bypass(id=891)

            target_field = 'target'

            # Get features and target
            X = dataset.data.features
            y = dataset.data.targets

            # Combine features and target
            X[target_field] = y

            return X, target_field

        except Exception as e:
            raise ValueError(f"Error loading UCI ML demo dataset: {str(e)}")
