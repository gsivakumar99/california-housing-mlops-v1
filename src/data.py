import os
from typing import Tuple

import dvc.api
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# DVC remote storage path
DATA_PATH = "data/california_housing.csv"


def save_data() -> None:
    """Save California Housing dataset to CSV for DVC tracking."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["PRICE"] = housing.target

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)


def load_data() -> pd.DataFrame:
    """Load data from DVC-tracked CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing housing data with features and target price
    """
    try:
        # Try to load from DVC-tracked file
        with dvc.api.open(DATA_PATH) as f:
            df = pd.read_csv(f)
    except Exception:
        # If file doesn't exist, create it
        save_data()
        df = pd.read_csv(DATA_PATH)

    return df


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess the housing data for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw housing data including features and target

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Tuple containing (X_train, X_test, y_train, y_test)
    """
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
