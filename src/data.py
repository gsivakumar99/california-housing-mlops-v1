from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data(enhanced: bool = False) -> pd.DataFrame:
    """Load California Housing dataset.

    Parameters
    ----------
    enhanced : bool, default=False
        Whether to include derived features

    Returns
    -------
    pd.DataFrame
        DataFrame containing housing data with features and target price
    """
    # Load base dataset
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["PRICE"] = housing.target

    # Add derived features if requested
    if enhanced:
        df["ROOMS_PER_HOUSEHOLD"] = df["AveRooms"] / df["AveOccup"]
        df["BEDROOMS_RATIO"] = df["AveBedrms"] / df["AveRooms"]

    return df


def preprocess_data(
    df: pd.DataFrame
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
    # Separate features and target
    y = df["PRICE"]
    
    # Select appropriate features
    feature_cols = [col for col in df.columns if col != "PRICE"]
    X = df[feature_cols]

    return train_test_split(X, y, test_size=0.2, random_state=42)
