import json
import os
from datetime import datetime

import pandas as pd
from sklearn.datasets import fetch_california_housing


def create_data_version() -> None:
    """Create a new version of the dataset with derived features."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Load base data
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["PRICE"] = housing.target

    # Add version info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df["VERSION"] = timestamp

    # Add derived features
    df["ROOMS_PER_HOUSEHOLD"] = df["AveRooms"] / df["AveOccup"]
    df["BEDROOMS_RATIO"] = df["AveBedrms"] / df["AveRooms"]

    # Validate data
    assert not df.isnull().any().any(), "Data contains null values"
    assert len(df) > 0, "Dataset is empty"
    assert all(col in df.columns for col in housing.feature_names)
    assert "PRICE" in df.columns, "Missing target variable"

    # Save data with version info
    df.to_csv("data/california_housing.csv", index=False)

    # Save version info
    version_info = {
        "version": timestamp,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "features": df.columns.tolist(),
        "has_derived_features": True,
        "derived_features": ["ROOMS_PER_HOUSEHOLD", "BEDROOMS_RATIO"],
        "created_at": datetime.now().isoformat()
    }

    with open("data/version_info.json", "w") as f:
        json.dump(version_info, f, indent=2)

    print("Data preparation successful")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Version: {timestamp}")


if __name__ == "__main__":
    create_data_version()
