import numpy as np
import pandas as pd
import matplotlib.pyplot as p

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from pyspikelib import TrainNormalizeTransform
from pyspikelib import TsfreshVectorizeTransform
from pyspikelib import TsfreshFeaturePreprocessorPipeline
from pyspikelib.utils import simple_undersampling
import pyspikelib.mpladeq as mpla

from examples.dataset_adapters import retina_dataset

from pathlib import Path

# %%
datapath = './data/retina/mode_paper_data'
retinal_spike_data = retina_dataset(datapath)
print(retinal_spike_data.keys())
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
def prepare_tsfresh_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=200, step=200)
    X, y = normalizer.transform(X, y, delimiter=None)
    return X, y


X_train, y_train = prepare_tsfresh_data(X_train, y_train)
X_test, y_test = prepare_tsfresh_data(X_test, y_test)
# %%
thr = 4e4
fstate, mstate = X_train[y_train == 0], X_train[y_train == 1]
p.hist(fstate[fstate < thr].flatten(), bins=100, alpha=0.8, density=True)
p.hist(mstate[mstate < thr].flatten(), bins=100, alpha=0.8, density=True)
mpla.prettify((10, 8))
# %%
vectorizer = TsfreshVectorizeTransform(feature_set=None)
X_train_ts = vectorizer.transform(X_train)
X_test_ts = vectorizer.transform(X_test)
# %%

preprocessing = TsfreshFeaturePreprocessorPipeline(
    do_scaling=True, remove_low_variance=True
).construct_pipeline()
preprocessing.fit(X_train_ts)
X_train = preprocessing.transform(X_train_ts)
X_test = preprocessing.transform(X_test_ts)
# %%
X_train.shape
# %%
def random_forest_scores(X_train, X_test, y_train, y_test, subsample_size=6000):
    X_train, y_train = simple_undersampling(
        X_train, y_train, subsample_size=subsample_size, pandas=False
    )
    X_test, y_test = simple_undersampling(
        X_test, y_test, subsample_size=subsample_size, pandas=False
    )
    # print('Target mean | train: {} test: {}'.format(y_train.mean(), y_test.mean()))
    # print('Dataset shape | train: {} test: {}'.format(X_train.shape, X_test.shape))
    forest = RandomForestClassifier(n_estimators=200, random_state=41, n_jobs=-1)
    forest.fit(X_train, y_train)
    acc_score = accuracy_score(y_test, forest.predict(X_test))
    auc_roc_score = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
    print('Accuracy & AUC-ROC scores of RF: {}, {}'.format(acc_score, auc_roc_score))
    return {'accuracy': acc_score, 'auc-roc': auc_roc_score}


# %%
trials = 20
for trial in range(trials):
    random_forest_scores(X_train, X_test, y_train, y_test, subsample_size=4000)

# %%
feature_names = [
    'abs_energy',
    'mean',
    'median',
    'minimum',
    'maximum',
    'standard_deviation',
]
simple_baseline_features = ['value__' + name for name in feature_names]
X_train = X_train.loc[:, simple_baseline_features]
X_test = X_test.loc[:, simple_baseline_features]
# %%
trials = 20
for trial in range(trials):
    random_forest_scores(X_train, X_test, y_train, y_test, subsample_size=4000)
