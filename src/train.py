import os
from datetime import datetime
from typing import Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, r2_score

from src.data import load_data, preprocess_data
from src.model import HousePriceModel

# Create model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                               "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_model(
    model: HousePriceModel,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, float]:
    """Evaluate model performance.

    Parameters
    ----------
    model : HousePriceModel
        Trained model
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Test features
    y_train : np.ndarray
        Training targets
    y_test : np.ndarray
        Test targets

    Returns
    -------
    Tuple[float, float, float]
        Train MSE, test MSE, and test R2 score
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    return train_mse, test_mse, test_r2


def train_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    experiment_name: str = "california_housing",
) -> HousePriceModel:
    """Train model with MLflow tracking.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in random forest
    max_depth : int, default=10
        Maximum depth of trees
    experiment_name : str, default="california_housing"
        Name of MLflow experiment

    Returns
    -------
    HousePriceModel
        Trained model
    """
    mlflow.set_experiment(experiment_name)

    # Create timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_dir = os.path.join(MODEL_DIR, f"model_{timestamp}")
    os.makedirs(model_version_dir, exist_ok=True)

    with mlflow.start_run():
        # Load and preprocess data
        df = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = HousePriceModel(n_estimators=n_estimators, 
                                    max_depth=max_depth)
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        train_mse, test_mse, test_r2 = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )

        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_r2", test_r2)

        # Create model signature and save
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:1]

        model_path = os.path.join(model_version_dir, "model.pkl")
        mlflow.sklearn.save_model(
            model,
            model_path,
            signature=signature,
            input_example=input_example,
        )

        # Update latest model symlink
        latest_path = os.path.join(MODEL_DIR, "latest")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_version_dir, latest_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
        )

        print(f"Model saved in: {model_version_dir}")
        print(f"Latest model symlink: {latest_path}")

        return model


if __name__ == "__main__":
    # Train multiple models with different parameters
    experiments = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 150, "max_depth": 12},
    ]

    for params in experiments:
        train_model(**params)
