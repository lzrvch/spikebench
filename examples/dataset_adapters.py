import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


def pasteur_dataset(datapath):
    spikes = {'groups': [], 'series': []}
    for file in glob(str(datapath) + '/*/Results*'):
        spike_trains = Path(file).read_text()
        for train in spike_trains.split('\n'):
            spikes['groups'].append(file + train.split(',')[0])
            spikes['series'].append(
                np.diff([float(val) for val in train.split(',')[1:]])
            )
    return spikes_dict_to_df(spikes)


def allen_dataset(datapath):
    loaded_spikes = pickle.load(open(datapath, 'rb'))
    multiplier = 1e8
    spikes = {'groups': [], 'series': []}
    for experiment in loaded_spikes:
        for neuron_id in experiment:
            if experiment[neuron_id]:
                for train in experiment[neuron_id]:
                    if train.any():
                        spikes['series'].append(
                            multiplier * np.diff(np.array(train, dtype=np.float64))
                        )
                        spikes['groups'].append(neuron_id)
    return spikes_dict_to_df(spikes)


def fcx1_dataset(file):
    spikes = {'groups': [], 'series': []}

    data = pd.read_parquet(file)
    for neuron_id in data.columns.values:
        spikes['groups'].append(neuron_id)
        series = [float(value) for value in data[neuron_id].values[0].split()]
        spikes['series'].append(np.array(series))
    return spikes_dict_to_df(spikes)


def spikes_dict_to_df(spikes_dict):
    stringify = lambda array: '{}'.format(list(array))[1:-1]

    spikes = [
        (spikes_dict['groups'][index], stringify(value))
        for index, value in enumerate(spikes_dict['series'])
        if value.any()
    ]

    spikes_dict['groups'] = [value[0] for value in spikes]
    spikes_dict['series'] = [value[1] for value in spikes]

    return pd.DataFrame(spikes_dict)
