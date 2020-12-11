import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .rtf_reader import rtf_to_text


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


def pasteur_calcium_dataset(datapath):
    calcium_traces = {'groups': [], 'series': [], 'timespan': []}
    for animal_directory in glob(str(datapath) + '/*/'):
        calcium_traces['groups'].append(animal_directory)
        metadata_file = glob(animal_directory + '*parameters*')
        if not metadata_file:
            timespan = None
        else:
            with open(metadata_file[0], 'r') as f:
                for line in f.readlines():
                    if 'rtf' in metadata_file[0]:
                        line = rtf_to_text(line)
                    else:
                        pass
                    if 't' in line:
                        timestamp = line.split('=')[1]
                        if 'sec' in timestamp:
                            timestamp = timestamp.split('sec')[0]
                        timespan = float(timestamp)
        calcium_traces['timespan'].append(timespan)
        traces_df = []
        for file in glob(animal_directory + '*xls'):
            traces_df.append(pd.read_excel(file, delimiter='\t', index_col=0, encoding = 'utf-8'))
        for file in glob(animal_directory + '*csv'):
            traces = pd.read_csv(file, delimiter='\t', index_col=0)
            if traces.shape[1] == 0:
                traces = pd.read_csv(file, delimiter=',', index_col=0)
            traces_df.append(traces)

        calcium_traces['series'].append(traces_df)
    return calcium_traces


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
    def stringify(array):
        return '{}'.format(list(array))[1:-1]

    spikes = [
        (spikes_dict['groups'][index], stringify(value))
        for index, value in enumerate(spikes_dict['series'])
        if value.any()
    ]

    spikes_dict['groups'] = [value[0] for value in spikes]
    spikes_dict['series'] = [value[1] for value in spikes]

    return pd.DataFrame(spikes_dict)


def retina_dataset(datapath):
    stimuli_types = [
        'randomly_moving_bar',
        'repeated_natural_movie',
        'unique_natural_movie',
        'white_noise_checkerboard',
    ]
    data = {}
    for key in stimuli_types:
        data[key] = loadmat(datapath + '/{}/data.mat'.format(key))

    spike_data = {}
    min_spike_count = 10

    for stimulus_type in stimuli_types:
        spike_data[stimulus_type] = {'groups': [], 'series': []}
        num_neurons = np.squeeze(data[stimulus_type]['data'][0, 0][2][0, 0][1]).shape[0]
        for neuron_index in range(num_neurons):
            neuron_isis = np.diff(
                np.squeeze(
                    np.squeeze(data[stimulus_type]['data'][0, 0][2][0, 0][1])[
                        neuron_index
                    ]
                )
            )
            if neuron_isis.shape[0] > min_spike_count:
                spike_data[stimulus_type]['groups'].append(neuron_index)
                spike_data[stimulus_type]['series'].append(
                    ' '.join(['{:.2f}'.format(value) for value in neuron_isis])
                )
        spike_data[stimulus_type] = pd.DataFrame(spike_data[stimulus_type])
    return spike_data
