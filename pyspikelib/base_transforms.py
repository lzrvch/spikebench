from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class NoFitMixin:
    def fit(self, X, y=None):
        return self


class DFTransform(TransformerMixin, NoFitMixin):
    def __init__(self, func, copy=False, **kwargs):
        self.func = func
        self.copy = copy
        self.kwargs = kwargs

    def set_params(self, **params):
        for key, value in params.items():
            self.kwargs[key] = value

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_, **self.kwargs)


class DFStandardScaler(TransformerMixin, NoFitMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        X[X.columns] = self.scaler.transform(X[X.columns])
        return X


class DFLowVarianceRemoval(TransformerMixin, NoFitMixin):
    def __init__(self, variance_threshold=0.2):
        self.high_variance_features = None
        self.variance_threshold = variance_threshold
        self.safety_eps = 1e-9

    def fit(self, X, y=None):
        self.high_variance_features = X.loc[
            :, (X.std() / (self.safety_eps + X.mean())).abs() > self.variance_threshold
        ].columns.values
        return self

    def transform(self, X):
        return X.loc[:, self.high_variance_features]
