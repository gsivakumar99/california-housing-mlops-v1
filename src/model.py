from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


# ML model
class HousePriceModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth
        }
