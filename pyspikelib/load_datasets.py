import os
import os.path
from copy import deepcopy
from pickle import NONE
from pathlib import Path

import gdown
import gzip
import shutil
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import pyspikelib.transforms as transforms
from pyspikelib.dataset_adapters import fcx1_dataset, retina_dataset, allen_dataset
from pyspikelib.encoders import ISIShuffleTransform


def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


def download_fcx1(data_path):
    data_path = Path(data_path)
    URL = 'https://drive.google.com/uc?id=1fQKpYPHmenob692YZaG1P7YKWCYaTw19'
    output = str(data_path / 'data.tar.gz')
    gdown.download(URL, output, quiet=False)
    shutil.unpack_archive(output, data_path)
    gunzip_shutil(str(data_path / 'wake.parq.gz'), str(data_path / 'wake.parq'))
    gunzip_shutil(str(data_path / 'sleep.parq.gz'), str(data_path / 'sleep.parq'))
    os.remove(output)
    os.remove(str(data_path / 'wake.parq.gz'))
    os.remove(str(data_path / 'sleep.parq.gz'))


def load_fcx1(dataset_path='./data/fcx1', random_seed=0, test_size=0.3,
    n_samples=None, window_size=200, step_size=100):
    DELIMITER = ','
    dataset_path = Path(dataset_path)

    if not os.path.exists(dataset_path):
        dataset_path.mkdir(parents=True, exist_ok=True)
        download_fcx1(dataset_path)

    wake_spikes = fcx1_dataset(dataset_path / 'wake.parq')
    sleep_spikes = fcx1_dataset(dataset_path / 'sleep.parq')

    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    X = np.hstack([wake_spikes.series.values, sleep_spikes.series.values])
    y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(sleep_spikes.shape[0])])
    groups = np.hstack([wake_spikes.groups.values, sleep_spikes.groups.values])

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    normalizer = transforms.TrainNormalizeTransform(
        window=window_size, step=step_size, n_samples=n_samples
    )
    X_train, y_train, groups_train = normalizer.transform(X_train, y_train, delimiter=DELIMITER)
    X_test, y_test, groups_test = normalizer.transform(X_test, y_test, delimiter=DELIMITER)
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def download_retina(data_path):
    data_path = Path(data_path)

    URL = 'https://drive.google.com/uc?id=1HqZSs7r14bC97gWvw_VJ63CsVl3Ug6DM'
    output = str(data_path / 'retinal_data.zip')
    gdown.download(URL, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    with zipfile.ZipFile(str(data_path / 'error_robust_mode_data.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(output)
    os.remove(str(data_path / 'error_robust_mode_data.zip'))


def load_retina(dataset_path='./data/retina', state1='randomly_moving_bar',
    state2='white_noise_checkerboard', random_seed=0, test_size=0.3, n_samples=None,
    window_size=200, step_size=200):

    dataset_path = Path(dataset_path)

    if not os.path.exists(dataset_path):
        dataset_path.mkdir(parents=True, exist_ok=True)
        download_retina(dataset_path)

    retinal_spike_data = retina_dataset(str(dataset_path / 'mode_paper_data'))
    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size,
        random_state=random_seed)
    X = np.hstack(
        [
            retinal_spike_data[state1].series.values,
            retinal_spike_data[state2].series.values,
        ]
    )
    y = np.hstack(
        [
            np.ones(retinal_spike_data[state1].shape[0]),
            np.zeros(retinal_spike_data[state2].shape[0]),
        ]
    )
    groups = np.hstack(
        [
            retinal_spike_data[state1].groups.values,
            retinal_spike_data[state2].groups.values,
        ]
    )

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    normalizer = transforms.TrainNormalizeTransform(
        window=window_size, step=step_size, n_samples=n_samples
    )
    X_train, y_train, groups_train = normalizer.transform(X_train, y_train, delimiter=None)
    X_test, y_test, groups_test = normalizer.transform(X_test, y_test, delimiter=None)
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def download_allen(data_path):
    data_path = Path(data_path)

    URL = 'https://drive.google.com/uc?id=1G4QWDTShP8C5uv5iwcZw4MFR-DClOG5Z'
    output = str(data_path / 'allen_data.zip')
    gdown.download(URL, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(output)


def load_allen(dataset_path='./data/allen', random_seed=0, test_size=0.3,
    n_samples=None, window_size=50, step_size=20):
    DELIMITER = ','
    dataset_path = Path(dataset_path)

    if not os.path.exists(dataset_path):
        dataset_path.mkdir(parents=True, exist_ok=True)
        download_allen(dataset_path)

    vip_datapath = Path(dataset_path) / 'Vip_spikes.pkl'
    vip_spike_data = allen_dataset(vip_datapath)
    sst_datapath = Path(dataset_path) / 'Sst_spikes.pkl'
    sst_spike_data = allen_dataset(sst_datapath)

    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    X = np.hstack([vip_spike_data.series.values, sst_spike_data.series.values])
    y = np.hstack(
        [np.ones(vip_spike_data.shape[0]), np.zeros(sst_spike_data.shape[0])]
    )
    groups = np.hstack([vip_spike_data.groups.values, sst_spike_data.groups.values])

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    normalizer = transforms.TrainNormalizeTransform(
        window=window_size, step=step_size, n_samples=n_samples
    )
    X_train, y_train, groups_train = normalizer.transform(X_train, y_train, delimiter=DELIMITER)
    X_test, y_test, groups_test = normalizer.transform(X_test, y_test, delimiter=DELIMITER)
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def load_fcx1_temporal(dataset_path='./data/fcx1', random_seed=0, test_size=0.3,
    n_samples=None, window_size=200, step_size=100):
    DELIMITER = ','
    dataset_path = Path(dataset_path)

    if not os.path.exists(dataset_path):
        dataset_path.mkdir(parents=True, exist_ok=True)
        download_fcx1(dataset_path)

    wake_spikes = fcx1_dataset(dataset_path / 'wake.parq')

    shuffler = ISIShuffleTransform()
    wake_spikes_shuffled = shuffler.transform(
        deepcopy(wake_spikes), format='pandas', delimiter=DELIMITER
    )

    group_split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    X = np.hstack([wake_spikes.series.values, wake_spikes_shuffled.series.values])
    y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(wake_spikes_shuffled.shape[0])])
    groups = np.hstack([wake_spikes.groups.values, wake_spikes_shuffled.groups.values])

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    normalizer = transforms.TrainNormalizeTransform(
        window=window_size, step=step_size, n_samples=n_samples
    )
    X_train, y_train, groups_train = normalizer.transform(X_train, y_train, delimiter=DELIMITER)
    X_test, y_test, groups_test = normalizer.transform(X_test, y_test, delimiter=DELIMITER)
    return X_train, X_test, y_train, y_test, groups_train, groups_test
