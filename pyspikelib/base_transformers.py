# PySpikeLib: A set of tools for neuronal spiking data mining
# Copyright (c) 2020 Ivan Lazarevich.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.

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
