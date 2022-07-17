import logging
import sys
import warnings
from dataclasses import asdict
from functools import partial

import chika
import pandas as pd
import spiketrainn as spknn
from pyspikelib import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from pyspikelib.helpers import set_random_seed, subsampled_fit_predict
from sklearn.neighbors import KNeighborsClassifier

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}

METRICS = ('isi', 'spike', 'schreiber', 'van_rossum', 'max_metric', 'modulus_metric')
knn_params = {
    'knn_k1_victor_purpura': {'n_neighbors': 1, 'metric': partial(spknn.distance, metric='victor_purpura', q=1.0)},
}
for metric_name in METRICS:
    knn_params[f'knn_k1_{metric_name}'] = {'n_neighbors': 1, 'metric': partial(spknn.distance, metric=metric_name)}

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
        raw_results.to_csv(f'./csv/{cfg.dataset}_raw_spk_knn_balanced.csv')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
