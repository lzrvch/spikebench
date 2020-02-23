from pathlib import Path

import matplotlib.pyplot as p
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

import pyspikelib.mpladeq as mpladeq
from examples.dataset_adapters import fcx1_dataset
from experiments.model_tuning.h2o_runner import H2ORunner
from pyspikelib import TrainNormalizeTransform
from pyspikelib.utils import simple_undersampling

mpladeq.beautify_mpl()
# %%
dataset = Path('./data/')
wake_spikes = fcx1_dataset(dataset / 'wake.parq')
sleep_spikes = fcx1_dataset(dataset / 'sleep.parq')

folds_num = 3
group_split = GroupKFold(n_splits=folds_num)
X = np.hstack([wake_spikes.series.values, sleep_spikes.series.values])
y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(sleep_spikes.shape[0])])
groups = np.hstack([wake_spikes.groups.values, sleep_spikes.groups.values])

X_fold, y_fold = {}, {}
fold_index = 0
for train_index, test_index in group_split.split(X, y, groups):
    X_fold[fold_index] = pd.DataFrame(
        {'series': X[test_index], 'groups': groups[test_index]}
    )
    y_fold[fold_index] = y[test_index]
    fold_index += 1


# %%
def prepare_isi_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=150, step=150, n_samples=3000)
    X, y = normalizer.transform(X, y, delimiter=',')
    return X, y


for fold_idx in range(folds_num):
    X_fold[fold_idx], y_fold[fold_idx] = prepare_isi_data(
        X_fold[fold_idx], y_fold[fold_idx]
    )
    X_fold[fold_idx], y_fold[fold_idx] = simple_undersampling(
        X_fold[fold_idx], y_fold[fold_idx], pandas=False
    )

# %%
X_fold[0].shape, X_fold[1].shape, X_fold[2].shape, y_fold[0].mean(), y_fold[
    1
].mean(), y_fold[2].mean()
# %%
assert np.sum((X_fold[0].values[:, None] == X_fold[1].values).all(-1).any(-1)) == 0
assert np.sum((X_fold[0].values[:, None] == X_fold[2].values).all(-1).any(-1)) == 0

# %% simple baseline
forest = RandomForestClassifier(n_estimators=200, random_state=41, n_jobs=-1)
forest.fit(X_fold[0], y_fold[0])
acc_score = accuracy_score(y_fold[2], forest.predict(X_fold[2]))
auc_roc_score = roc_auc_score(y_fold[2], forest.predict_proba(X_fold[2])[:, 1])
print(
    'Accuracy & AUC-ROC scores of baseline RF: {}, {}'.format(acc_score, auc_roc_score)
)
# %% prepare h2o calculation
runner = H2ORunner(nthreads=-1, max_mem_size=12, logdir='./h2o_runs_dump')
runner.set_data(
    train_data=(X_fold[0], y_fold[0]),
    valid_data=(X_fold[1], y_fold[1]),
    test_data=(X_fold[2], y_fold[2]),
)
runner.run(max_runtime_secs=10000)
lb = runner.leaderboard
# %%
lb.shape
lb.head(20)
# %%
p.scatter(lb.auc, 1 - lb.mean_per_class_error)
mpladeq.prettify((10, 8))
