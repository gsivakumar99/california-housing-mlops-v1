from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class HousePriceModel(BaseEstimator, RegressorMixin):
    """House Price Prediction Model using Random Forest Regressor.

    This model predicts house prices based on various features such as median
    income, house age, average rooms, etc.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int, default=10
        The maximum depth of each tree.

    Attributes
    ----------
    model : RandomForestRegressor
        The underlying sklearn random forest model.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10) -> None:
        """Initialize the model with given parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HousePriceModel":
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features
        y : array-like of shape (n_samples,)
            Target house prices

        Returns
        -------
        self : HousePriceModel
            The fitted model
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to make predictions for

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted house prices
        """
        return self.model.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters of the model.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }
