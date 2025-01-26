import os

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.data import load_data, preprocess_data
from src.model import HousePriceModel


@pytest.fixture
def model_dir():
    """Create model directory for tests."""
    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "latest"
    )
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    return X_train, X_test, y_train, y_test


def test_data_loading():
    """Test if data loading function works correctly."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "PRICE" in df.columns
    assert len(df.columns) == 9  # 8 features + 1 target


def test_data_preprocessing(sample_data):
    """Test if data preprocessing function works correctly."""
    X_train, X_test, y_train, y_test = sample_data

    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train.shape) == 1
    assert len(y_test.shape) == 1
    # Fix: Check dtypes instead of dtype
    assert X_train.dtypes.all() == np.float64
    assert y_train.dtype == np.float64
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()


def test_model_training(sample_data):
    """Test model training and prediction functionality."""
    X_train, X_test, y_train, y_test = sample_data

    model = HousePriceModel()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert isinstance(predictions, np.ndarray)
    assert all(isinstance(pred, np.float64) for pred in predictions)
    assert all(pred >= 0 for pred in predictions)


def test_model_parameters():
    """Test model parameter handling."""
    model = HousePriceModel(n_estimators=50, max_depth=5)
    params = model.get_params()

    assert params["n_estimators"] == 50
    assert params["max_depth"] == 5


def test_untrained_model():
    """Test behavior of untrained model."""
    model = HousePriceModel()
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.fixture
def client():
    """Create test client."""
    from src.predict import create_app
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint(client):
    """Test the prediction endpoint."""
    # Test with valid input
    test_input = {
        "features": [
            8.3252, 41.0, 6.984127, 1.023810, 322.0,
            2.555556, 37.88, -122.23
        ]
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "prediction" in response.get_json()

    # Test with invalid input
    invalid_input = {
        "features": [1, 2]  # Wrong number of features
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 400


def test_info_endpoint(client):
    """Test the info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    assert "feature_names" in response.get_json()
