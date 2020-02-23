import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from addict import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from examples.dataset_adapters import fcx1_dataset
from pyspikelib.fit_predict import tsfresh_fit_predict
from pyspikelib.train_encoders import ISIShuffleTransform

warnings.filterwarnings('ignore')
# %%
run_config = {
    'window': 200,
    'step': 200,
    'scale': True,
    'remove_low_variance': True,
    'trials': 10,
    'delimiter': ',',
    'train_subsample_factor': 0.7,
    'test_subsample_factor': 0.7,
}
run_config = Dict(run_config)
# %%
dataset = Path('./data/')
wake_spikes = fcx1_dataset(dataset / 'wake.parq')
shuffler = ISIShuffleTransform()
wake_spikes_shuffled = shuffler.transform(
    deepcopy(wake_spikes), format='pandas', delimiter=run_config.delimiter
)
# %%
group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
X = np.hstack([wake_spikes.series.values, wake_spikes_shuffled.series.values])
y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(wake_spikes_shuffled.shape[0])])
groups = np.hstack([wake_spikes.groups.values, wake_spikes_shuffled.groups.values])

for train_index, test_index in group_split.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})
# %%
forest = RandomForestClassifier(n_estimators=200, random_state=41, n_jobs=-1)
tsfresh_fit_predict(forest, X_train, X_test, y_train, y_test, run_config)
