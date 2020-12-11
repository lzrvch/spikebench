import numpy as np
import pandas as pd

import allensdk.core.json_utilities as json_utilities

from addict import Dict
from tqdm import tqdm

from allensdk.api.queries.glif_api import GlifApi
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.model.glif.glif_neuron import GlifNeuron

from experiments.model_generated_data.colorednoise import (
    powerlaw_psd_gaussian as colorednoise,
)


def generate_train_from_model_id(model_id, stimulus_amplitude=1e-8,
                                 duration=1e6, noise_exponent=0):
    glif_api = GlifApi()
    neuron_config = glif_api.get_neuron_configs([model_id])
    neuron = GlifNeuron.from_dict(neuron_config[model_id])
    stimulus = stimulus_amplitude * colorednoise(exponent=noise_exponent,
                                                 size=int(duration))
    neuron.dt = 5e-6
    output = neuron.run(stimulus)
    spike_times = output['interpolated_spike_times']
    return spike_times


def model_ids_for_cell_type(cells_df, cell_type_tag):
    model_ids = []
    type_cells = cells_df[cells_df.transgenic_line.str.contains(cell_type_tag)]
    glif_api = GlifApi()
    for neuron_id in type_cells.index.values:
        if glif_api.get_neuronal_models([neuron_id]):
            models_metadata = glif_api.get_neuronal_models([neuron_id])[0]
            for model in models_metadata['neuronal_models']:
                model_ids.append(model['id'])
    return np.array(model_ids)


if __name__ == '__main__':
    config = Dict({
        'cell_type_tag': 'Vip',
        'stimulus_amplitude': 1e-8,
        'duration': 5e6,
        'noise_exponent': 1,
        'dump_file': './experiments/model_generated_data/data/vip_glif_spike_trains_exp1.csv'
    })

    ctc = CellTypesCache()
    cells = ctc.get_cells(species=[CellTypesApi.MOUSE], require_reconstruction=False)
    cells_df = pd.DataFrame.from_records(cells, index='id')

    model_id_list = model_ids_for_cell_type(cells_df, config.cell_type_tag)
    spike_trains = []
    for model_id in tqdm(model_id_list):
        spike_train = generate_train_from_model_id(model_id,
                                            stimulus_amplitude=config.stimulus_amplitude,
                                     duration=config.duration,
                                     noise_exponent=config.noise_exponent)
        spike_trains.append(' '.join(['{:2f}'.format(value) for value in spike_train]))
        pd.Series(spike_trains).to_csv(config.dump_file)
