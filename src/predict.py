import logging
import os
from typing import Any, Dict

import mlflow
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.datasets import fetch_california_housing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get the model directory path
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models"
)
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "latest", "model.pkl")

# Get feature names
feature_names = fetch_california_housing().feature_names
if not isinstance(feature_names, list):
    feature_names = feature_names.tolist()


def load_model() -> Any:
    """Load the trained model or train a new one if none exists.

    Returns
    -------
    Any
        Loaded or newly trained model
    """
    try:
        model = mlflow.sklearn.load_model(LATEST_MODEL_PATH)
        logger.info(f"Loaded model from: {LATEST_MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Training new model...")
        from src.train import train_model
        return train_model()


def validate_features(features: list) -> bool:
    """Validate input features.

    Parameters
    ----------
    features : list
        List of input features to validate

    Returns
    -------
    bool
        True if features are valid, False otherwise
    """
    if not isinstance(features, list):
        return False
    if len(features) != len(feature_names):
        return False
    if not all(isinstance(x, (int, float)) for x in features):
        return False
    return True


# Load the model at startup
model = load_model()


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, str]:
    """Check if the service is healthy.

    Returns
    -------
    Dict[str, str]
        Health status
    """
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """Make predictions using the loaded model.

    Returns
    -------
    Dict[str, Any]
        Prediction results or error message
    """
    try:
        # Get and validate input
        if not request.is_json:
            raise ValueError("Request must be JSON")

        features = request.json.get("features")
        if not validate_features(features):
            raise ValueError(
                f"Invalid features. Expected {len(feature_names)} "
                "numeric values"
            )

        # Create DataFrame and make prediction
        df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(df)

        return jsonify({
            "status": "success",
            "prediction": float(prediction[0]),
            "feature_names": feature_names
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route("/info", methods=["GET"])
def info() -> Dict[str, Any]:
    """Get information about expected features.

    Returns
    -------
    Dict[str, Any]
        Model and feature information
    """
    return jsonify({
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
            "Longitude": "Block group longitude"
        },
        "example_input": {
            "features": [
                8.3252, 41.0, 6.984127, 1.023810, 322.0,
                2.555556, 37.88, -122.23
            ]
        }
    })


@app.route("/version", methods=["GET"])
def model_version() -> Dict[str, Any]:
    """Get model version information.

    Returns
    -------
    Dict[str, Any]
        Model version details
    """
    try:
        model_info = mlflow.sklearn.get_model_info(LATEST_MODEL_PATH)
        return jsonify({
            "status": "success",
            "model_path": LATEST_MODEL_PATH,
            "model_info": {
                "run_id": model_info.run_id,
                "artifact_path": model_info.artifact_path,
                "utc_time_created": str(model_info.utc_time_created)
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error getting model version: {str(e)}"
        }), 400


def create_app() -> Flask:
    """Create and configure the Flask app.

    Returns
    -------
    Flask
        Configured Flask application
    """
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
