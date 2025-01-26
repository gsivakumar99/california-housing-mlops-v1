import os

import numpy as np
import pandas as pd
import pytest

from src.data import load_data, preprocess_data


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
    df = load_data(enhanced=False)  # Use base dataset for tests
    X_train, X_test, y_train, y_test = preprocess_data(df)
    return X_train, X_test, y_train, y_test


def test_data_loading():
    """Test if data loading function works correctly."""
    # Test base dataset
    df = load_data(enhanced=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "PRICE" in df.columns
    assert len(df.columns) == 9  # 8 features + 1 target

    # Test enhanced dataset
    df_enhanced = load_data(enhanced=True)
    assert len(df_enhanced.columns) == 11  # 8 base + 1 target + 2 derived


def test_data_preprocessing(sample_data):
    """Test if data preprocessing function works correctly."""
    X_train, X_test, y_train, y_test = sample_data

    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train.shape) == 1
    assert len(y_test.shape) == 1
    
    # Check each column's dtype
    assert all(dtype == np.float64 for dtype in X_train.dtypes)
    assert y_train.dtype == np.float64
    
    # Check for missing values
    assert not X_train.isnull().any().any()
    assert not X_test.isnull().any().any()


@pytest.fixture
def client():
    """Create test client."""
    from src.predict import create_app, ensure_model_loaded

    # Initialize the app in test mode
    app = create_app(testing=True)
    
    # Ensure model is loaded before tests
    with app.app_context():
        ensure_model_loaded()
    
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
    data = response.get_json()
    assert "prediction" in data
    assert data["status"] == "success"

    # Test with invalid input
    invalid_input = {
        "features": [1, 2]  # Wrong number of features
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_info_endpoint(client):
    """Test the info endpoint."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.get_json()
    assert "feature_names" in data
    assert "description" in data
