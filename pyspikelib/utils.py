import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import tsfresh.utilities.dataframe_functions as tsfresh_utils


def loadtxt(file, limit=float('inf'), ids_present=True, do_differencing=True):
    spike_data = {}
    spike_data['series'] = []
    spike_data['ids'] = []

    with open(file) as outfile:
        for index, line in enumerate(outfile):
            clean_line = [x for x in line.strip().split(',') if len(x)]
            if ids_present:
                train_id = clean_line[0]
                clean_line = np.array(clean_line[1:]).astype(float)
                if do_differencing:
                    clean_line = np.diff(clean_line)
                if len(clean_line) > 0:
                    spike_data['series'].append(clean_line)
                    spike_data['ids'].append(train_id)
            else:
                if len(clean_line) > 0:
                    clean_line = np.array(clean_line).astype(float)
                    if do_differencing:
                        clean_line = np.diff(clean_line)
                    spike_data['series'].append(clean_line)
                    spike_data['ids'].append(index)
            if index > limit:
                break

    return spike_data


def save_spikes_to_parquet(spike_data, filename, num_digits=3):
    data = {}
    for index, key in enumerate(spike_data['ids']):
        data[key] = [' '.join([str(round(value, num_digits)) for value
                               in spike_data['series'][index]]), ]

    pd.DataFrame(data).to_parquet(filename)


def load_parquet(file):
    spike_data = {}
    spike_data['ids'] = []
    spike_data['series'] = []

    data = pd.read_parquet(file)
    for neuron_id in data.columns.values:
        spike_data['ids'].append(neuron_id)
        series = [float(value) for value in data[neuron_id].values[0].split()]
        spike_data['series'].append(np.array(series))

    return spike_data


def split_by_spikes(spike_data, ratio=0.5):
    train_lengths = [len(train) for train in spike_data['series']]

    total_spikes = np.sum(train_lengths)

    cum_length = 0
    for index, train_length in enumerate(train_lengths):
        cum_length += train_length
        if cum_length > int(ratio * total_spikes):
            split_index = index
            break

    first_spike_data = {}
    second_spike_data = {}
    first_spike_data['series'] = spike_data['series'][:split_index]
    first_spike_data['ids'] = spike_data['ids'][:split_index]
    second_spike_data['series'] = spike_data['series'][split_index:]
    second_spike_data['ids'] = spike_data['ids'][split_index:]

    return first_spike_data, second_spike_data


def crop_isi_samples(
        isi_series_list,
        window_size=50,
        step_size=50,
        upper_isi_thr=float('inf'),
        upper_percentage=1.0,
        lower_isi_thr=-1e6,
        lower_percentage=1.0,
        total_samples=None,
        shuffle_isis=False,
        sampling_rate=1,
        condition=None):
    isi_samples = []
    sample_ids = []
    sample_freqs = []
    samples = {}

    if condition is None:
        def condition(series): return True

    for train_index, spike_train in enumerate(isi_series_list['series']):
        for index in range(
                0, int(
                    np.floor(
                        spike_train.shape[0] / step_size - 1))):
            train_sample = spike_train[index *
                                       step_size:(index * step_size + window_size)]
            sample_freqs.append(train_sample.mean())
            if ((train_sample > lower_isi_thr).mean() >= lower_percentage) & (
                    (train_sample < upper_isi_thr).mean() >= upper_percentage):
                if train_sample.shape[0] == window_size:
                    if shuffle_isis:
                        train_sample = pd.Series(train_sample).sample(
                            n=train_sample.shape[0]).values
                    if condition(train_sample):
                        sample_ids.append(isi_series_list['ids'][train_index])
                        isi_samples.append(train_sample[::sampling_rate])

    isi_samples = np.array(isi_samples)
    sample_ids = np.array(sample_ids)

    if total_samples:
        if total_samples > isi_samples.shape[0]:
            print(
                'WARNING: total_samples (%i) larger than the number of samples (%i)!' %
                (total_samples, isi_samples.shape[0]))

        chosen_indices = np.random.choice(isi_samples.shape[0], total_samples)
        isi_samples = isi_samples[chosen_indices, :]
        sample_ids = sample_ids[chosen_indices]

    samples['series'] = isi_samples
    samples['ids'] = sample_ids

    return samples


def tsfresh_vectorize(data_array, to_file=None,
                      feature_dict=None, n_jobs=8,
                      verbose=True):
    df = pd.DataFrame(columns=['id', 'time', 'value'], dtype=float)
    for index in range(data_array.shape[1]):
        tmp = pd.DataFrame(data_array[:, index],
                           columns=['value'])
        tmp['id'] = list(range(data_array.shape[0]))
        tmp['time'] = [index] * data_array.shape[0]
        df = pd.concat([df, tmp],
                       ignore_index=True, sort=False)

    if feature_dict:
        if feature_dict == 'simple_baseline':
            ts_feature_dict = {
                'abs_energy': None,
                'mean': None,
                'median': None,
                'minimum': None,
                'maximum': None,
                'standard_deviation': None
            }
        elif feature_dict == 'no_entropy':
            ts_feature_dict = ComprehensiveFCParameters()
            entropy_keys = [key for key in ts_feature_dict.keys()
                            if 'entropy' in key]
            for key in entropy_keys:
                ts_feature_dict.pop(key, None)
        elif feature_dict == 'distribution_features':
            ts_feature_dict = distribution_features_tsfresh_dict()
        elif feature_dict == 'temporal_features':
            full_feature_dict = ComprehensiveFCParameters()
            distro_features = distribution_features_tsfresh_dict()
            ts_feature_dict = {key: full_feature_dict[key] for key in set(
                full_feature_dict) - set(distro_features)}
        else:
            ts_feature_dict = feature_dict
    else:
        ts_feature_dict = ComprehensiveFCParameters()

    X_feats = extract_features(df,
                               default_fc_parameters=ts_feature_dict,
                               column_id='id',
                               column_sort='time',
                               disable_progressbar=np.logical_not(verbose),
                               n_jobs=n_jobs)

    if to_file:
        X_feats.to_csv(to_file, sep=',')

    return X_feats


def preprocess_tsfresh_features(features_df, impute=True,
                                do_scaling=True, scaler=None,
                                remove_low_variance=True,
                                keep_features_list=None):
    if impute:
        tsfresh_utils.impute(features_df)

    if do_scaling:

        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)
        else:
            X_scaled = scaler.transform(features_df)

        X_scaled = pd.DataFrame(X_scaled,
                                index=features_df.index,
                                columns=features_df.columns)
    else:
        X_scaled = features_df

    if remove_low_variance:
        X_scaled = X_scaled.loc[:, (X_scaled.std(
        ) / (1e-9 + X_scaled.mean())).abs() > 0.2]

    # Todo correlation removal routine

    if keep_features_list:
        X_final = X_scaled.loc[:, keep_features_list]
    else:
        X_final = X_scaled

    return X_final, scaler


def train_test_common_features(train_dataframe, test_dataframe):
    train_feature_set = set(train_dataframe.columns.values)
    test_feature_set = set(test_dataframe.columns.values)

    train_dataframe = train_dataframe.loc[:, train_feature_set.intersection(
        test_feature_set)]
    test_dataframe = test_dataframe.loc[:, train_feature_set.intersection(
        test_feature_set)]

    return train_dataframe, test_dataframe


def cross_validation_tsfresh(tsdata, model, train_test_names,
                             metric_score=None,
                             importance_measure='forest',
                             return_model=False,
                             samples=None,
                             test_samples=None,
                             trials=5):
    scores = []
    importance = {}

    if not samples:
        samples = tsdata[train_test_names[0]].shape[0]

    if not test_samples:
        test_samples = samples

    def forest_importances(model):
        return model.feature_importances_

    def accuracy(target, prediction, prediction_proba):
        return accuracy_score(target, prediction)

    if importance_measure == 'forest':
        importance_measure = forest_importances

    if not metric_score:
        metric_score = accuracy

    for index in range(trials):
        indices = np.random.choice(
            tsdata[train_test_names[0]].shape[0], samples)
        test_indices = np.random.choice(
            tsdata[train_test_names[2]].shape[0], test_samples)

        X_train = pd.concat([tsdata[train_test_names[0]].iloc[indices, :],
                             tsdata[train_test_names[1]].iloc[indices, :]])
        y_train = np.array([0] * indices.shape[0] + [1] * indices.shape[0])

        X_test = pd.concat([tsdata[train_test_names[2]].iloc[test_indices, :],
                            tsdata[train_test_names[3]].iloc[test_indices, :]])
        y_test = np.array(
            [0] *
            test_indices.shape[0] +
            [1] *
            test_indices.shape[0])

        X_train, scaler = preprocess_tsfresh_features(X_train,
                                                      remove_low_variance=True)

        X_test, _ = preprocess_tsfresh_features(X_test,
                                                scaler=scaler,
                                                remove_low_variance=True)

        X_train, X_test = train_test_common_features(X_train, X_test)

        model.fit(X_train, y_train)

        score = metric_score(y_test,
                             model.predict(X_test),
                             model.predict_proba(X_test)[:, 1])

        scores.append(score)

        importance['feature_%i' % index] = X_train.columns.values
        importance['importance_%i' % index] = importance_measure(model)

    if not return_model:
        return scores, importance
    else:
        total_samples = tsdata[train_test_names[0]].shape[0]
        X_train = pd.concat([tsdata[train_test_names[0]],
                             tsdata[train_test_names[1]]])
        y_train = np.array([0] * total_samples + [1] * total_samples)

        X_test = pd.concat([tsdata[train_test_names[2]],
                            tsdata[train_test_names[3]]])
        y_test = np.array([0] * total_samples + [1] * total_samples)

        X_train, scaler = preprocess_tsfresh_features(X_train,
                                                      remove_low_variance=True)

        X_test, _ = preprocess_tsfresh_features(X_test,
                                                scaler=scaler,
                                                remove_low_variance=True)

        X_train, X_test = train_test_common_features(X_train, X_test)

        model.fit(X_train, y_train)

        return scores, importance, model


def model_evaluation_pipe(data, model, tsfresh_mode, names, window_size,
                          step_size, total_samples, sub_samples, trials=5):
    crop_data = {}

    for key in data:
        crop_data[key] = crop_isi_samples(data[key],
                                          window_size=window_size,
                                          step_size=step_size,
                                          total_samples=total_samples)

    tsdata = {}
    for key in crop_data:
        tsdata[key] = tsfresh_vectorize(crop_data[key]['series'],
                                        feature_dict=tsfresh_mode,
                                        n_jobs=24)

    scores, importance = cross_validation_tsfresh(tsdata=tsdata,
                                                  model=model,
                                                  train_test_names=names,
                                                  samples=sub_samples,
                                                  trials=trials)

    return scores, importance


def distribution_features_tsfresh_dict():
    ratios_beyond_r_sigma_rvalues = [1, 1.5, 2, 2.5,
                                     3, 5, 6, 7, 10]

    feature_dict = {
        'symmetry_looking': [{'r': value} for value
                             in np.arange(0.05, 1.0, 0.05)],
        'standard_deviation': None,
        'kurtosis': None,
        'variance_larger_than_standard_deviation': None,
        'ratio_beyond_r_sigma': [{'r': value} for value in ratios_beyond_r_sigma_rvalues],
        'count_below_mean': None,
        'maximum': None,
        'variance': None,
        'abs_energy': None,
        'mean': None,
        'skewness': None,
        'length': None,
        'large_standard_deviation': [{'r': value} for value
                                     in np.arange(0.05, 1.0, 0.05)],
        'count_above_mean': None,
        'minimum': None,
        'sum_values': None,
        'quantile': [{'q': value} for value
                     in np.arange(0.1, 1.0, 0.1)],
        'ratio_value_number_to_time_series_length': None,
        'median': None}

    return feature_dict


def tsfresh_dataframe_stats(df):
    unique_values = []

    for key in df.columns.values:
        unique_values.append(
            pd.Series(
                df[key].values.astype(
                    np.float32)).value_counts().values.shape[0])

    unique_values = np.array(unique_values)

    max_values = 30
    features = {}
    features['nan'] = df.columns.values[np.where(unique_values == 0)[0]]
    features['binary'] = df.columns.values[np.where(unique_values == 2)[0]]
    features['categorial'] = df.columns.values[np.where(
        (unique_values > 2) & (unique_values < max_values))[0]]

    return features

# ToDo cross-validation sklearn subclass based on neuron ID splitting
