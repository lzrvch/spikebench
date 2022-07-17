import logging
import sys
import warnings
from dataclasses import asdict
from functools import partial

import chika
import pandas as pd
import scipy.stats as stats
from pyspikelib import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from pyspikelib.helpers import set_random_seed, subsampled_fit_predict
from pyts.metrics import dtw
from sklearn.neighbors import KNeighborsClassifier

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}


def ks_distance(sample1, sample2):
    return stats.ks_2samp(sample1, sample2)[0]

def em_distance(sample1, sample2):
    return stats.wasserstein_distance(sample1, sample2)

def dtw_distance(sample1, sample2, radius=50):
    return dtw(sample1, sample2, method='fast', options={'radius': radius})


knn_params = {
    'knn_k1_lp1': {'n_neighbors': 1, 'p': 1},
    'knn_k5_lp2': {'n_neighbors': 5, 'p': 2},
    'knn_k5_lp1': {'n_neighbors': 5, 'p': 1},
    'knn_k5_lp1_wd': {'n_neighbors': 5, 'p': 1, 'weights': 'distance'},
    'knn_k1_ks': {'n_neighbors': 1, 'metric': ks_distance},
    'knn_k1_em': {'n_neighbors': 1, 'metric': em_distance},
    'knn_k1_dtw_r50': {'n_neighbors': 1, 'metric': partial(dtw_distance, radius=50)},
    'knn_k1_dtw_r25': {'n_neighbors': 1, 'metric': partial(dtw_distance, radius=25)},
    'knn_k1_dtw_r10': {'n_neighbors': 1, 'metric': partial(dtw_distance, radius=10)},
}

model_zoo = {name: KNeighborsClassifier(**model_params, n_jobs=-1) for
    name, model_params in knn_params.items()}


@chika.config
class Config:
    seed: int = 0
    dataset: str = 'retina'
    train_subsample_factor: float = 0.7
    test_subsample_factor: float = 0.7
    trials: int = 10


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    loader_fn = DATASET_NAME_LOADER_MAP[cfg.dataset]
    X_train, X_test, y_train, y_test, gr_train, gr_test = loader_fn(random_seed=cfg.seed)

    logging.info(
        f'Dataset shape after preprocessing: train {X_train.shape}, test {X_test.shape}'
    )
    logging.info(
        f'Mean target values: train {y_train.mean()}, test {y_test.mean()}'
    )

    raw_results = pd.DataFrame()
    for model_name, model in model_zoo.items():
        results = subsampled_fit_predict(model, X_train, X_test, y_train, y_test, cfg, predict_train=False)
        results['model'] = ['raw_' + model_name] * cfg.trials
        raw_results = pd.concat([raw_results, results], axis=0)
        raw_results.to_csv(f'./csv/{cfg.dataset}_raw_knn_balanced.csv')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
