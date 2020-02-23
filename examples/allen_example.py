import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from addict import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from examples.dataset_adapters import allen_dataset
from pyspikelib.fit_predict import tsfresh_fit_predict

warnings.filterwarnings('ignore')
# %%
np.random.seed(0)

run_config = {
    'window': 50,
    'step': 20,
    'scale': True,
    'remove_low_variance': True,
    'trials': 10,
    'delimiter': ',',
    'train_subsample_factor': 0.7,
    'test_subsample_factor': 0.7,
}
# %%
datapath = Path('./data/allen/Vip_spikes_dict_new.pkl')
vip_spike_data = allen_dataset(datapath)
datapath = Path('./data/allen/Sst_spikes_dict_new.pkl')
sst_spike_data = allen_dataset(datapath)
# %%
group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
X = np.hstack([vip_spike_data.series.values, sst_spike_data.series.values])
y = np.hstack([np.ones(vip_spike_data.shape[0]), np.zeros(sst_spike_data.shape[0]), ])
groups = np.hstack([vip_spike_data.groups.values, sst_spike_data.groups.values])

for train_index, test_index in group_split.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})
# %%
forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
tsfresh_fit_predict(forest, X_train, X_test, y_train, y_test, Dict(run_config))
