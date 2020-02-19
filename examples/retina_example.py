import warnings

import numpy as np
import pandas as pd
from addict import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from tsfresh_baseline import tsfresh_fit_predict

from examples.dataset_adapters import retina_dataset

warnings.filterwarnings('ignore')
# %%
run_config = {
    'window': 200,
    'step': 200,
    'scale': False,
    'remove_low_variance': False,
    'trials': 10,
    'delimiter': None,
    'train_subsample_factor': 0.7,
    'test_subsample_factor': 0.7,
}
# %%
datapath = './data/retina/mode_paper_data'
retinal_spike_data = retina_dataset(datapath)

fstate, mstate = 'randomly_moving_bar', 'white_noise_checkerboard'
# fstate, mstate = 'repeated_natural_movie', 'unique_natural_movie'
# %%
group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
X = np.hstack(
    [retinal_spike_data[fstate].series.values,
     retinal_spike_data[mstate].series.values]
)
y = np.hstack(
    [
        np.ones(retinal_spike_data[fstate].shape[0]),
        np.zeros(retinal_spike_data[mstate].shape[0]),
    ]
)
groups = np.hstack(
    [retinal_spike_data[fstate].groups.values,
     retinal_spike_data[mstate].groups.values]
)

for train_index, test_index in group_split.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})
# %%
forest = RandomForestClassifier(n_estimators=200, random_state=41, n_jobs=-1)
tsfresh_fit_predict(forest, X_train, X_test, y_train, y_test, Dict(run_config))
