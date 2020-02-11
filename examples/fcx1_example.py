import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pyspikelib.train_transformers import TrainNormalizeTransform
from pyspikelib.train_transformers import TsfreshVectorizeTransform
from pyspikelib.train_transformers import TsfreshFeaturePreprocessorPipeline
from pyspikelib.utils import train_test_common_features
from dataset_adapters import fcx1_dataset

from pathlib import Path

# %%
dataset = Path('./data/')
wake_spikes = fcx1_dataset(dataset / 'wake.parq')
sleep_spikes = fcx1_dataset(dataset / 'sleep.parq')

group_kfold = GroupKFold(n_splits=2)
X = np.hstack([wake_spikes.series.values, sleep_spikes.series.values])
y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(sleep_spikes.shape[0])])
groups = np.hstack([wake_spikes.groups.values, sleep_spikes.groups.values])

for train_index, test_index in group_kfold.split(X, y, groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

# %%
def prepare_tsfresh_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=50, step=50, n_samples=1000)
    X, y = normalizer.transform(X, y, delimiter=',')
    vectorizer = TsfreshVectorizeTransform(feature_set='distribution_features')
    X = vectorizer.transform(X)
    return X, y


X_train, y_train = prepare_tsfresh_data(X_train, y_train)
X_test, y_test = prepare_tsfresh_data(X_test, y_test)

preprocessing = TsfreshFeaturePreprocessorPipeline().construct_pipeline()
preprocessing.fit(X_train)
X_train = preprocessing.transform(X_train)
X_test = preprocessing.transform(X_test)

X_train, X_test = train_test_common_features(X_train, X_test)
# %%
forest = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
# %%
print('AUC-ROC score of RF: {}'.format(roc_auc_score(y_test, forest.predict(X_test))))
