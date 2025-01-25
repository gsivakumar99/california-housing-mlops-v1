import os
from datetime import datetime

import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, r2_score

from data import load_data, preprocess_data
from model import HousePriceModel

# Create model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(n_estimators=100, max_depth=10):
    mlflow.set_experiment("california_housing")

    # Create a timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_dir = os.path.join(MODEL_DIR, f'model_{timestamp}')
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

        # Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Log metrics
        mlflow.log_metric("train_mse", mean_squared_error(y_train,
                                                          train_pred))
        mlflow.log_metric("test_mse", mean_squared_error(y_test,
                                                         test_pred))
        mlflow.log_metric("test_r2", r2_score(y_test, test_pred))

        # Create model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Create an input example
        input_example = X_train.iloc[:1]

        # Save versioned model
        model_path = os.path.join(model_version_dir, 'model.pkl')
        mlflow.sklearn.save_model(
            model,
            model_path,
            signature=signature,
            input_example=input_example
        )

        # Create a symlink to the latest model
        latest_path = os.path.join(MODEL_DIR, 'latest')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_version_dir, latest_path)

        # Also log model to MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example
        )

        print(f"Model saved in: {model_version_dir}")
        print(f"Latest model symlink: {latest_path}")

        return model


if __name__ == "__main__":
    train_model()
