from functools import partial

import numpy as np
import pandas as pd
import psutil
import tsfresh.utilities.dataframe_functions as tsfresh_utils
from sklearn.base import TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

import spikebench.helpers as helpers
from spikebench.base_transforms import (DFLowVarianceRemoval, DFStandardScaler,
                                        DFTransform, NoFitMixin)

CPU_COUNT = psutil.cpu_count(logical=True)


class TrainNormalizeTransform(TransformerMixin, NoFitMixin):
    def __init__(self, window=20, step=20, n_samples=None):
        self.window = window
        self.step = step
        self.n_samples = n_samples

    @staticmethod
    def string_to_float_series(string_series, delimiter=None):
        return np.array([float(value) for value in string_series.split(delimiter)])

    @staticmethod
    def rolling_window(a, window, step):
        n_chunks = (a.shape[0] - window) // step + 1
        split_chunks = np.array(
            [np.roll(a, -step * index)[:window] for index in range(n_chunks)]
        )
        if split_chunks.any():
            return np.vstack(split_chunks)

    def transform(self, X, y=None, delimiter=None):
        normalized_trains = np.zeros(self.window)
        target = np.array([])
        groups = np.array([])
        for train_index, spike_train in enumerate(X.series.values):
            spike_train = self.string_to_float_series(spike_train, delimiter=delimiter)
            split_chunks = self.rolling_window(
                spike_train, window=self.window, step=self.step
            )
            if split_chunks is not None:
                normalized_trains = np.vstack([normalized_trains, split_chunks])
                target = np.append(target, [y[train_index]] * split_chunks.shape[0])
                groups = np.append(groups, [X.groups[train_index]] * split_chunks.shape[0])

        normalized_trains = normalized_trains[1:, :]
        if self.n_samples is not None:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=self.n_samples)
            for train_index, test_index in splitter.split(normalized_trains, target):
                normalized_trains = normalized_trains[train_index, :]
                target = target[train_index]
                groups = groups[train_index]
        return np.vstack(normalized_trains), target, groups


class TsfreshVectorizeTransform(TransformerMixin, NoFitMixin):
    def __init__(self, to_file=None, feature_set=None, n_jobs=CPU_COUNT, verbose=True):
        self.to_file = to_file
        self.feature_set = feature_set
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
        return df

    @staticmethod
    def get_feature_dict(feature_set=None):
        full_feature_dict = ComprehensiveFCParameters()
        simple_baseline_features = {
            key: None
            for key in [
                'abs_energy',
                'mean',
                'median',
                'minimum',
                'maximum',
                'standard_deviation',
            ]
        }
        distribution_features_dict = helpers.distribution_features_tsfresh_dict()
        temporal_feature_dict = {
            key: full_feature_dict[key]
            for key in set(full_feature_dict) - set(distribution_features_dict)
        }
        no_entropy_features_dict = {
            key: value
            for key, value in full_feature_dict.items()
            if 'entropy' not in key
        }
        feature_dict = {
            'simple_baseline': simple_baseline_features,
            'distribution_features': distribution_features_dict,
            'temporal_features': temporal_feature_dict,
            'no_entropy': no_entropy_features_dict,
            'full': full_feature_dict,
        }
        return feature_dict.get(feature_set, full_feature_dict)

    def transform(self, X):
        tsfresh_df = self.transform_to_tsfresh_format(X)
        ts_feature_dict = self.get_feature_dict(self.feature_set)
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
                        partial(_select_features, feature_list=self.keep_features_list)
                    ),
                )
            )
        if self.impute:
            chained_transformers.append(
                ('imputation', DFTransform(_tsfresh_imputation))
            )
        if self.remove_low_variance:
            chained_transformers.append(('low_var_removal', DFLowVarianceRemoval()))
        if self.do_scaling:
            chained_transformers.append(('standard_scaling', DFStandardScaler()))
        # TODO: add correlation removal step
        return Pipeline(chained_transformers)
