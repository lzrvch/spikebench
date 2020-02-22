import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import torch.utils.data as data_utils

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, AUCCallback, F1ScoreCallback

from sklearn.model_selection import GroupShuffleSplit

from examples.dataset_adapters import fcx1_dataset
from experiments.neural_nets.models import CNN1DNet

from pyspikelib import TrainNormalizeTransform

warnings.filterwarnings('ignore')
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
def prepare_isi_data(X, y):
    """Extract and preprocess tsfresh features from spiking data"""
    normalizer = TrainNormalizeTransform(window=100, step=100)
    X, y = normalizer.transform(X, y, delimiter=',')
    return X, y


X_train, y_train = prepare_isi_data(X_train, y_train)
X_test, y_test = prepare_isi_data(X_test, y_test)
# %%
X_train_torch = torch.from_numpy(X_train).float().cuda()
X_test_torch = torch.from_numpy(X_test).float().cuda()
y_train_torch = torch.from_numpy(y_train).long().cuda()
y_test_torch = torch.from_numpy(y_test).long().cuda()

train = data_utils.TensorDataset(X_train_torch, y_train_torch)
train_loader = data_utils.DataLoader(train, batch_size=256, shuffle=True)
val = data_utils.TensorDataset(X_test_torch, y_test_torch)
val_loader = data_utils.DataLoader(val, batch_size=256, shuffle=False)

logdir = "./catalyst_runs/1d_cnn_5000_epochs"
num_epochs = 5000
loaders = {'train': train_loader, 'valid': val_loader}


model = CNN1DNet(input_size=1, num_classes=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

runner = SupervisedRunner()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    # scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    callbacks=[AccuracyCallback(num_classes=2)],
    main_metric='accuracy01',
    minimize_metric=False,
    verbose=True,
)
