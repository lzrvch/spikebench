import logging
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from tsfresh import extract_features

import spikebench.transforms as transforms


def distribution_features_tsfresh_dict():
    ratios_beyond_r_sigma_rvalues = [1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]

    feature_dict = {
        'symmetry_looking': [{'r': value} for value in np.arange(0.05, 1.0, 0.05)],
        'standard_deviation': None,
        'kurtosis': None,
        'variance_larger_than_standard_deviation': None,
        'ratio_beyond_r_sigma': [
            {'r': value} for value in ratios_beyond_r_sigma_rvalues
        ],
        'count_below_mean': None,
        'maximum': None,
        'variance': None,
        'abs_energy': None,
        'mean': None,
        'skewness': None,
        'length': None,
        'large_standard_deviation': [
            {'r': value} for value in np.arange(0.05, 1.0, 0.05)
        ],
        'count_above_mean': None,
        'minimum': None,
        'sum_values': None,
        'quantile': [{'q': value} for value in np.arange(0.1, 1.0, 0.1)],
        'ratio_value_number_to_time_series_length': None,
        'median': None,
    }

    return feature_dict


def tsfresh_dataframe_stats(df):
    unique_values = []

    for key in df.columns.values:
        unique_values.append(
            pd.Series(df[key].values.astype(np.float32)).value_counts().values.shape[0]
        )

    unique_values = np.array(unique_values)

    max_values = 30
    features = {}
    features['nan'] = df.columns.values[np.where(unique_values == 0)[0]]
    features['binary'] = df.columns.values[np.where(unique_values == 2)[0]]
    features['categorial'] = df.columns.values[
        np.where((unique_values > 2) & (unique_values < max_values))[0]
    ]

    return features


def train_test_common_features(train_df, test_df):
    train_feature_set = set(train_df.columns.values)
    test_feature_set = set(test_df.columns.values)
    train_df = train_df.loc[:, train_feature_set.intersection(test_feature_set)]
    test_df = test_df.loc[:, train_feature_set.intersection(test_feature_set)]
    return train_df, test_df


def simple_undersampling(
    X, y, subsample_size=None, pandas=False
):
    dominant_class_label = int(y.mean() > 0.5)
    X = pd.DataFrame(X) if not pandas else X
    num_samples = (y != dominant_class_label).sum()
    dominant_indices = np.random.choice(X.shape[0] - num_samples,
                                        num_samples, replace=False)
    X_undersampled = pd.concat(
        [
            X.iloc[np.where(y != dominant_class_label)[0], :],
            X.iloc[np.where(y == dominant_class_label)[0], :].iloc[dominant_indices, :],
        ]
    )
    y_undersampled = np.array([int(not dominant_class_label)] * num_samples + [dominant_class_label] * num_samples)
    if subsample_size is not None:
        sample_indices = np.random.choice(
            X_undersampled.shape[0],
            int(subsample_size * X_undersampled.shape[0]),
            replace=False,
        )
        X_undersampled, y_undersampled = (
            X_undersampled.iloc[sample_indices, :],
            y_undersampled[sample_indices],
        )
    return X_undersampled.values, y_undersampled


def tsfresh_vectorize(X_train, X_test, y_train, y_test, config, cache_file=None):
    if cache_file is not None and Path(cache_file).exists():
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    else:
        logging.info('Started time series vectorization and preprocessing')

        vectorizer = transforms.TsfreshVectorizeTransform(feature_set=config.tsfresh_feature_set)
        X_train = vectorizer.transform(X_train)
        X_test = vectorizer.transform(X_test)

        preprocessing = transforms.TsfreshFeaturePreprocessorPipeline(
            do_scaling=config.tsfresh_scale_features, remove_low_variance=config.tsfresh_remove_low_variance
        ).construct_pipeline()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_test = preprocessing.transform(X_test)

        if cache_file is not None:
            with open(cache_file, 'wb') as f:
                pickle.dump((X_train, y_train, X_test, y_test), f)

    return X_train, X_test, y_train, y_test


def tsfresh_vectorize_spike_count(X_train, X_test, y_train, y_test, config, cache_file=None):
    if cache_file is not None and Path(cache_file).exists():
        with open(cache_file, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    else:
        logging.info('Started time series vectorization and preprocessing')

        def extract_tsfresh_feats(X):
            X = X.series.apply(lambda row: np.array([float(v) for v in row.split()]).astype(np.float32))
            df_train = pd.DataFrame(columns=['id', 'time', 'value'], dtype=float)

            for idx, ts in tqdm(enumerate(X)):
                tmp = pd.DataFrame(ts, columns=['value'])
                tmp['id'] = [idx] * ts.shape[0]
                tmp['time'] = list(range(ts.shape[0]))
                df_train = pd.concat([df_train, tmp], ignore_index=True, sort=False)
            return extract_features(
                df_train,
                column_id='id',
                column_sort='time',
                disable_progressbar=False,
                n_jobs=psutil.cpu_count(logical=True),
            )

        X_train = extract_tsfresh_feats(X_train)
        X_test = extract_tsfresh_feats(X_test)

        preprocessing = transforms.TsfreshFeaturePreprocessorPipeline(
            do_scaling=config.tsfresh_scale_features, remove_low_variance=config.tsfresh_remove_low_variance
        ).construct_pipeline()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_test = preprocessing.transform(X_test)

        if cache_file is not None:
            with open(cache_file, 'wb') as f:
                pickle.dump((X_train, y_train, X_test, y_test), f)

    return X_train, X_test, y_train, y_test


def subsampled_fit_predict(models, X_train, X_test, y_train, y_test, config, predict_train=True):
    result_table_columns = ['model_name', 'trial', 'accuracy_test', 'auc_roc_test']
    if predict_train:
        result_table_columns += ['accuracy_train', 'auc_roc_train']
    results = {key: [] for key in result_table_columns}
    metrics_to_collect = {'accuracy': accuracy_score, 'auc_roc': roc_auc_score}

    for trial_idx in range(config.trials):
        logging.info(
            f'Training models on subsample # {trial_idx}/{config.trials}'
        )
        X_train_sample_balanced, y_train_sample_balanced = simple_undersampling(
            X_train, y_train, subsample_size=config.train_subsample_factor,
        )
        X_test_sample_balanced, y_test_sample_balanced = simple_undersampling(
            X_test, y_test, subsample_size=config.test_subsample_factor,
        )

        for model_name, model in models.items():
            logging.info(f'Fitting model {model_name}')
            model.fit(X_train_sample_balanced, y_train_sample_balanced)
            val_sets = [((X_test_sample_balanced, y_test_sample_balanced), 'test')]
            if predict_train:
                val_sets += [((X_train_sample_balanced, y_train_sample_balanced), 'train')]
            for (X, y), dataset_label in val_sets:
                for metric_name, metric_fn in metrics_to_collect.items():
                    model_predictions = model.predict(X) \
                        if metric_name not in ['auc_roc'] else model.predict_proba(X)[:, 1]
                    results[metric_name + '_' + dataset_label].append(metric_fn(y, model_predictions))
            results['trial'].append(trial_idx)
            results['model_name'].append(model_name)

    return pd.DataFrame(results).sort_values(by='model_name')


def set_random_seed(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
