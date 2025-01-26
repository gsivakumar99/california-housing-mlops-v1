import os
from typing import Any, Dict

import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

# Get the model directory path
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
)
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "latest", "model.pkl")

# Get feature names
feature_names = fetch_california_housing().feature_names
if not isinstance(feature_names, list):
    feature_names = feature_names.tolist()


def load_model():
    """Load the latest trained model."""
    try:
        model = mlflow.sklearn.load_model(LATEST_MODEL_PATH)
        print(f"Loaded model from: {LATEST_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run training first: python -m src.train")
        raise


# Load the model at startup
model = load_model()


@app.route("/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """Make predictions using the loaded model.

    Returns
    -------
    Dict[str, Any]
        Prediction results or error message
    """
    try:
        features = request.json["features"]
        df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(df)

        return jsonify(
            {
                "status": "success",
                "prediction": float(prediction[0]),
                "feature_names": feature_names,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/info", methods=["GET"])
def info() -> Dict[str, Any]:
    """Get information about expected features.

    Returns
    -------
    Dict[str, Any]
        Model and feature information
    """
    return jsonify(
        {
            "feature_names": feature_names,
            "description": "California Housing Price Prediction Model",
            "feature_descriptions": {
                "MedInc": "Median income in block group",
                "HouseAge": "Median house age in block group",
                "AveRooms": "Average number of rooms per household",
                "AveBedrms": "Average number of bedrooms per household",
                "Population": "Block group population",
                "AveOccup": "Average number of household members",
                "Latitude": "Block group latitude",
                "Longitude": "Block group longitude",
            },
            "example_input": {
                "features": [
                    8.3252,
                    41.0,
                    6.984127,
                    1.023810,
                    322.0,
                    2.555556,
                    37.88,
                    -122.23,
                ]
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
