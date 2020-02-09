import numpy as np
import pandas as pd
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import tsfresh.utilities.dataframe_functions as tsfresh_utils


class NoFitMixin:
    def fit(self, X, y=None):
        return self


class DFTransform(TransformerMixin, NoFitMixin):
    def __init__(self, func, copy=False):
        self.func = func
        self.copy = copy

    def transform(self, X):
        X_ = X if not self.copy else X.copy()
        return self.func(X_)


class TrainNormalizeTransform(TransformerMixin, NoFitMixin):
    def __init__(self, window=20, step=20, n_samples=1000):
        self.window = window
        self.step = step
        self.n_samples = n_samples

    @staticmethod
    def string_to_float_series(string_series, delimiter=','):
        return np.array([float(value) for value in string_series.split(delimiter)])

    @staticmethod
    def rolling_window(a, window, step):
        n_chunks = (a.shape[0] - window) // step + 1
        return np.vstack(
            [np.roll(a, -step * index)[:window] for index in range(n_chunks)]
        )

    def transform(self, X, delimiter=','):
        normalized_trains = []
        for spike_train in X.series.values:
            spike_train = self.string_to_float_series(spike_train, delimiter=delimiter)
            normalized_trains.append(
                self.rolling_window(spike_train, window=self.window, step=self.step)
            )
        return np.vstack(normalized_trains)


class TsfreshVectorizeTransform(TransformerMixin, NoFitMixin):
    def __init__(self, to_file=None, feature_dict=None, n_jobs=8, verbose=True):
        self.to_file = to_file
        self.feature_dict = feature_dict
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def transform_to_tsfresh_format(X):
        df = pd.DataFrame(columns=['id', 'time', 'value'], dtype=float)
        for index in range(X.shape[1]):
            tmp = pd.DataFrame(X[:, index], columns=['value'])
            tmp['id'] = list(range(X.shape[0]))
            tmp['time'] = [index] * X.shape[0]
            df = pd.concat([df, tmp], ignore_index=True, sort=False)

    @staticmethod
    def get_feature_dict(feature_dict=None):
        return ComprehensiveFCParameters()

    def transform(self, X):
        tsfresh_df = self.transform_to_tsfresh_format(X)
        ts_feature_dict = self.get_feature_dict(self.feature_dict)
        X_feats = extract_features(
            tsfresh_df,
            default_fc_parameters=ts_feature_dict,
            column_id='id',
            column_sort='time',
            disable_progressbar=np.logical_not(self.verbose),
            n_jobs=self.n_jobs,
        )
        return X_feats


def _tsfresh_imputation(X):
    tsfresh_utils.impute(X)
    return X


def _low_variance_removal(X):
    return X.loc[:, (X.std() / (1e-9 + X.mean())).abs() > 0.2]


def _select_features(X, feature_list=None):
    feature_list = X.columns.value if feature_list is None else feature_list
    return X.loc[:, feature_list]


class TsfreshFeaturePreprocessorPipeline:
    def __init__(
        self,
        impute=True,
        do_scaling=True,
        remove_low_variance=True,
        keep_features_list=None,
    ):
        self.impute = impute
        self.do_scaling = do_scaling
        self.remove_low_variance = remove_low_variance
        self.keep_features_list = keep_features_list

    def construct_pipeline(self):
        chained_transformers = []
        if self.keep_features_list is not None:
            chained_transformers.append(
                (
                    'select_features',
                    DFTransform(
                        partial(_select_features, feature_list=keep_features_list)
                    ),
                )
            )
        if self.impute is not None:
            chained_transformers.append(
                ('imputation', DFTransform(_tsfresh_imputation))
            )
        if self.do_scaling is not None:
            chained_transformers.append(('standard_scaling', StandardScaler))
        if self.remove_low_variance is not None:
            chained_transformers.append(
                ('low_var_removal', DFTransform(_low_variance_removal))
            )
        return Pipeline(chained_transformers)
