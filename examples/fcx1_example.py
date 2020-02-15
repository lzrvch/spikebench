import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from pyspikelib import TrainNormalizeTransform
from pyspikelib import TsfreshVectorizeTransform
from pyspikelib import TsfreshFeaturePreprocessorPipeline
from pyspikelib.utils import simple_undersampling

from dataset_adapters import fcx1_dataset

from pathlib import Path

# %%
dataset = Path('./data/')
wake_spikes = fcx1_dataset(dataset / 'wake.parq')
sleep_spikes = fcx1_dataset(dataset / 'sleep.parq')

group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
X = np.hstack([wake_spikes.series.values, sleep_spikes.series.values])
y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(sleep_spikes.shape[0])])
groups = np.hstack([wake_spikes.groups.values, sleep_spikes.groups.values])

for train_index, test_index in group_split.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

# %%
def prepare_tsfresh_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=150, step=150, n_samples=8000)
    X, y = normalizer.transform(X, y, delimiter=',')
    vectorizer = TsfreshVectorizeTransform(feature_set='simple_baseline')
    X = vectorizer.transform(X)
    return X, y


X_train_full, y_train = prepare_tsfresh_data(X_train, y_train)
X_test_full, y_test = prepare_tsfresh_data(X_test, y_test)
# %%
preprocessing = TsfreshFeaturePreprocessorPipeline(
    do_scaling=True, remove_low_variance=True
).construct_pipeline()
preprocessing.fit(X_train_full)
X_train = preprocessing.transform(X_train_full)
X_test = preprocessing.transform(X_test_full)
# %%
def random_forest_scores(X_train, X_test, y_train, y_test, subsample_size=5000):
    X_train, y_train = simple_undersampling(X_train, y_train, subsample_size=5000)
    X_test, y_test = simple_undersampling(X_test, y_test, subsample_size=5000)
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
    random_forest_scores(X_train, X_test, y_train, y_test, subsample_size=5000)
