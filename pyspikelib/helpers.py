import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import pyspikelib.transforms as transforms


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
    X, y, subsample_size=None, pandas=True
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
    return X_undersampled, y_undersampled


def tsfresh_fit_predict(model, X_train, X_test, y_train, y_test,
                        config, load_dataset_from=None, dump_dataset_to=None):

    if load_dataset_from is not None:
        with open(load_dataset_from, 'rb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    else:
        logging.info('Started time series vectorization and preprocessing')

        vectorizer = transforms.TsfreshVectorizeTransform(feature_set=config.feature_set)
        X_train = vectorizer.transform(X_train)
        X_test = vectorizer.transform(X_test)

        preprocessing = transforms.TsfreshFeaturePreprocessorPipeline(
            do_scaling=config.scale, remove_low_variance=config.remove_low_variance
        ).construct_pipeline()
        preprocessing.fit(X_train)
        X_train = preprocessing.transform(X_train)
        X_test = preprocessing.transform(X_test)

    logging.info(
        'Dataset shape after preprocessing: train {}, test {}'.format(X_train.shape, X_test.shape)
    )
    logging.info(
        'Average target value: train {}, test {}'.format(y_train.mean(), y_test.mean())
    )

    if dump_dataset_to is not None:
        with open(dump_dataset_to, 'wb') as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)

    result_table_columns = ['trial', 'feature_set',
                            'accuracy_test', 'auc_roc_test',
                            'accuracy_train', 'auc_roc_train']
    results = {key: [] for key in result_table_columns}
    baseline_feature_names = [
        'abs_energy',
        'mean',
        'median',
        'minimum',
        'maximum',
        'standard_deviation',
    ]
    baseline_feature_names = ['value__' + name for name
                              in baseline_feature_names]
    metrics_to_collect = {'accuracy': accuracy_score, 'auc_roc': roc_auc_score}

    logging.info(
        'Training classifiers on dataset subsamples for {} trials'.format(config.trials)
    )
    for trial_idx in range(config.trials):
        X_train_sample_balanced, y_train_sample_balanced = simple_undersampling(
            X_train, y_train, subsample_size=config.train_subsample_factor,
        )
        X_test_sample_balanced, y_test_sample_balanced = simple_undersampling(
            X_test, y_test, subsample_size=config.test_subsample_factor,
        )

        model.fit(X_train_sample_balanced, y_train_sample_balanced)
        for (X, y), dataset_label in [((X_test_sample_balanced, y_test_sample_balanced), 'test'),
                                      ((X_train_sample_balanced, y_train_sample_balanced), 'train')]:
            for metric_name, metric_fn in metrics_to_collect.items():
                model_predictions = model.predict(X) \
                    if metric_name not in ['auc_roc'] else model.predict_proba(X)[:, 1]
                results[metric_name + '_' + dataset_label].append(metric_fn(y, model_predictions))
        results['feature_set'].append(config.feature_set)
        results['trial'].append(trial_idx)

        X_train_sample_balanced = X_train_sample_balanced.loc[:, baseline_feature_names]
        X_test_sample_balanced = X_test_sample_balanced.loc[:, baseline_feature_names]

        model.fit(X_train_sample_balanced, y_train_sample_balanced)

        for (X, y), dataset_label in [((X_test_sample_balanced, y_test_sample_balanced), 'test'),
                                      ((X_train_sample_balanced, y_train_sample_balanced), 'train')]:
            for metric_name, metric_fn in metrics_to_collect.items():
                model_predictions = model.predict(X) \
                    if metric_name not in ['auc_roc'] else model.predict_proba(X)[:, 1]
                results[metric_name + '_' + dataset_label].append(metric_fn(y, model_predictions))

        results['feature_set'].append('simple_baseline')
        results['trial'].append(trial_idx)

    return pd.DataFrame(results)
