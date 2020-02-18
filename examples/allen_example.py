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

from examples.dataset_adapters import allen_dataset

from pathlib import Path

# %%
datapath = Path('./data/allen/Vip_spikes_dict_new.pkl')
vip_spike_data = allen_dataset(datapath)
datapath = Path('./data/allen/Sst_spikes_dict_new.pkl')
sst_spike_data = allen_dataset(datapath)
# %%
group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
X = np.hstack(
    [vip_spike_data.series.values, sst_spike_data.series.values]
)
y = np.hstack(
    [
        np.ones(vip_spike_data.shape[0]),
        np.zeros(sst_spike_data.shape[0]),
    ]
)
groups = np.hstack(
    [vip_spike_data.groups.values, sst_spike_data.groups.values]
)

for train_index, test_index in group_split.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})
# %%
def prepare_tsfresh_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=50, step=20)
    X, y = normalizer.transform(X, y, delimiter=',')
    vectorizer = TsfreshVectorizeTransform(feature_set=None)
    X = vectorizer.transform(X)
    return X, y


X_train_ts, y_train = prepare_tsfresh_data(X_train, y_train)
X_test_ts, y_test = prepare_tsfresh_data(X_test, y_test)
# %%
preprocessing = TsfreshFeaturePreprocessorPipeline(
    do_scaling=True, remove_low_variance=True
).construct_pipeline()
preprocessing.fit(X_train_ts)
X_train = preprocessing.transform(X_train_ts)
X_test = preprocessing.transform(X_test_ts)
print('{} {}'.format(X_train_ts.shape, X_test_ts.shape))
# %%
def random_forest_scores(X_train, X_test, y_train, y_test,
                         train_subsample_size=1500, test_subsample_size=1500):
    X_train, y_train = simple_undersampling(
        X_train, y_train, subsample_size=train_subsample_size, pandas=False
    )
    X_test, y_test = simple_undersampling(
        X_test, y_test, subsample_size=test_subsample_size, pandas=False
    )
    forest = RandomForestClassifier(n_estimators=200, random_state=41, n_jobs=-1)
    forest.fit(X_train, y_train)
    acc_score = accuracy_score(y_test, forest.predict(X_test))
    auc_roc_score = roc_auc_score(y_test, forest.predict_proba(X_test)[:, 1])
    print('Accuracy & AUC-ROC scores of RF: {}, {}'.format(acc_score, auc_roc_score))
    return {'accuracy': acc_score, 'auc-roc': auc_roc_score}
# %%
trials = 10
for trial in range(trials):
    random_forest_scores(X_train, X_test, y_train, y_test,
                         train_subsample_size=1500, test_subsample_size=1500)
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
print('SIMPLE BASELINE:')
trials = 10
for trial in range(trials):
    random_forest_scores(X_train, X_test, y_train, y_test, 
                         train_subsample_size=1500, test_subsample_size=1500)
