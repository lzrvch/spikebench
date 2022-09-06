import logging
import sys
import warnings
from dataclasses import asdict

import chika
import numpy as np
import pandas as pd
from fastai.callback.tracker import SaveModelCallback
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from spikebench import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from spikebench.helpers import set_random_seed
from tsai.all import *

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}

OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
}

model_zoo = {
    'inceptiontime_plus': InceptionTimePlus,
    'xceptionime_plus': XceptionTimePlus,
    'fcn_plus': FCNPlus,
    'resnet_plus': ResNetPlus,
}

@chika.config
class Config:
    seed: int = 0
    dataset: str = 'retina'
    preprocessing: bool = True
    epochs: int = 200
    lr: float = 1e-1
    optimizer: str = 'sgd'
    weight_decay: float = 1e-4
    batch_size: int = 128
    test_batch_size: int = 512
    workers: int = 0
    val_split: float = None


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    loader_fn = DATASET_NAME_LOADER_MAP[cfg.dataset]
    X_train, X_test, y_train, y_test, gr_train, gr_test = loader_fn(random_seed=cfg.seed)

    X_val, y_val = X_test, y_test
    if cfg.val_split is not None:
        group_split = GroupShuffleSplit(n_splits=1, test_size=cfg.val_split)
        for train_index, val_index in group_split.split(X_train, y_train, gr_train):
            X_train, X_val = X_train[train_index], X_train[val_index]
            y_train, y_val = y_train[train_index], y_train[val_index]


    if cfg.preprocessing:
        scaler = StandardScaler()
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)
        X_val = np.log1p(X_val)
        X_train = scaler.fit_transform(X_train)[:, np.newaxis, :]
        X_test = scaler.transform(X_test)[:, np.newaxis, :]
        X_val = scaler.transform(X_val)[:, np.newaxis, :]

    logging.info(
        f'Dataset shape after preprocessing: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}'
    )
    logging.info(
        f'Mean target values: train {y_train.mean()}, val {y_val.mean()}, test {y_test.mean()}'
    )

    models = {f'tsai_{model_name}': model for model_name, model in model_zoo.items()}

    X, y, splits = combine_split_data([X_train, X_val],
            [y_train, y_val])

    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

    test_dset = TSDatasets(X_test, y_test, tfms=tfms, inplace=True)

    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        test_dset,
        bs=[cfg.batch_size, cfg.test_batch_size, cfg.test_batch_size],
        batch_tfms=[TSStandardize()],
        num_workers=cfg.workers,
    )

    results = {'model_name': [], 'roc_auc': [], 'gmean': [], 'cohen_kappa': []}
    for model_name, model_cls in models.items():
        logging.info(f'Fitting model {model_name}')
        best_ckpt_name = f'best_model_{cfg.dataset}_{model_name}'

        model = model_cls(dls.vars, dls.c)
        learn = Learner(dls, model, metrics=CohenKappa(), opt_func=OPTIMIZERS[cfg.optimizer])
        callbacks = [SaveModelCallback(monitor='cohen_kappa_score', fname=best_ckpt_name)]

        # learn.fit(n_epoch=cfg.epochs, lr=cfg.lr)
        # learn.fit_one_cycle(n_epoch=cfg.epochs, lr_max=cfg.lr)
        learn.fit_flat_cos(n_epoch=cfg.epochs, lr=cfg.lr, wd=cfg.weight_decay, cbs=callbacks)

        learn.load(best_ckpt_name)
        valid_probas, valid_targets, valid_preds = learn.get_preds(
            dl=dls[-1], with_decoded=True
        )

        results['model_name'].append(model_name)
        results['roc_auc'].append(roc_auc_score(y_test, valid_probas[:, 1]))
        results['gmean'].append(geometric_mean_score(y_test, valid_preds))
        results['cohen_kappa'].append(cohen_kappa_score(y_test, valid_preds))

        res_table = pd.DataFrame(results)
        logging.info(f'Validation metrics\n{res_table}')
        res_table.to_csv(f'./csv/{cfg.dataset}_raw_tsai_imbalanced.csv', index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
