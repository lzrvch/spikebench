import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from pyspikelib import TrainNormalizeTransform
from pyspikelib import TsfreshFeaturePreprocessorPipeline
from pyspikelib import TsfreshVectorizeTransform
from pyspikelib.utils import simple_undersampling


def tsfresh_fit_predict(model, X_train, X_test, y_train, y_test,
                        config, load_dataset_from=None, dump_dataset_to=None):

    if load_dataset_from is not None:
        with open(load_dataset_from, 'wb') as f:
            X_train, y_train, X_test, y_test = pickle.load(f)
    else:
        logging.info('Started time series vectorization and preprocessing')
        X_train, y_train = tsfresh_vectorize(X_train, y_train, config)
        X_test, y_test = tsfresh_vectorize(X_test, y_test, config)
        preprocessing = TsfreshFeaturePreprocessorPipeline(
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
            X_train, y_train, subsample_size=config.train_subsample_factor
        )
        X_test_sample_balanced, y_test_sample_balanced = simple_undersampling(
            X_test, y_test, subsample_size=config.test_subsample_factor
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


def tsfresh_vectorize(X, y, config):
    n_samples = None if 'n_samples' not in config else config.n_samples
    normalizer = TrainNormalizeTransform(
        window=config.window, step=config.step, n_samples=n_samples
    )
    vectorizer = TsfreshVectorizeTransform(feature_set=config.feature_set)
    X, y = normalizer.transform(X, y, delimiter=config.delimiter)
    X = vectorizer.transform(X)
    return X, y
