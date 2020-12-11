import warnings

import numpy as np
import pandas as pd
from addict import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from examples.dataset_adapters import retina_dataset
from pyspikelib.fit_predict import tsfresh_fit_predict
from pyspikelib import TrainNormalizeTransform
from pyspikelib.utils import simple_undersampling

from pyts.classification import BOSSVS, SAXVSM, LearningShapelets

warnings.filterwarnings('ignore')
# %%
np.random.seed(13)

run_config = {
    'window': 50,
    'step': 50,
    'delimiter': None,
}
run_config = Dict(run_config)
# %%
datapath = './data/retina/mode_paper_data'
retinal_spike_data = retina_dataset(datapath)

# fstate, mstate = 'randomly_moving_bar', 'white_noise_checkerboard'
fstate, mstate = 'repeated_natural_movie', 'unique_natural_movie'
# %%
group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
X = np.hstack(
    [retinal_spike_data[fstate].series.values, retinal_spike_data[mstate].series.values]
)
y = np.hstack(
    [
        np.ones(retinal_spike_data[fstate].shape[0]),
        np.zeros(retinal_spike_data[mstate].shape[0]),
    ]
)
groups = np.hstack(
    [retinal_spike_data[fstate].groups.values, retinal_spike_data[mstate].groups.values]
)

for train_index, test_index in group_split.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})
# %%
def prepare_isi_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=run_config.window, step=run_config.step)
    X, y = normalizer.transform(X, y, delimiter=run_config.delimiter)
    return X, y


X_train, y_train = prepare_isi_data(X_train, y_train)
X_test, y_test = prepare_isi_data(X_test, y_test)
# %%
X_train.shape, X_test.shape, y_train.mean(), y_test.mean()
# %%
X_train, y_train = simple_undersampling(
    X_train, y_train, subsample_size=0.2, pandas=False
)
X_test, y_test = simple_undersampling(X_test, y_test, subsample_size=0.2, pandas=False)
# %%
classifier = BOSSVS(window_size=30)
classifier.fit(X_train, y_train)
# %%
accuracy_score(y_test, classifier.predict(X_test))
# %%
classifier = SAXVSM(window_size=20, sublinear_tf=False, use_idf=False)
classifier.fit(X_train, y_train)
# %%
accuracy_score(y_test, classifier.predict(X_test))
# %%
classifier = LearningShapelets(random_state=0, tol=0.1)
classifier.fit(X_train, y_train)
# %%
accuracy_score(y_test, classifier.predict(X_test))
# %%
from pyts.transformation import ROCKET

rocket = ROCKET(n_kernels=500, random_state=42)
X_train = rocket.fit_transform(X_train)
X_test = rocket.fit_transform(X_test)
# %%
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(X_train, y_train)
accuracy_score(y_test, forest.predict(X_test))
