import logging
import sys
import warnings
from dataclasses import asdict
from functools import partial

import chika
import numpy as np
import pandas as pd
import spiketrainn as spknn
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from spikebench import load_allen, load_fcx1, load_retina, load_temporal
from spikebench.helpers import set_random_seed, subsampled_fit_predict

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_temporal,
}


METRICS = ('isi', 'spike', 'schreiber', 'van_rossum', 'max_metric', 'modulus_metric')
knn_params = {
    'knn_k1_victor_purpura': {'n_neighbors': 1, 'metric': partial(spknn.distance, metric='victor_purpura', q=1.0)},
}
for metric_name in METRICS:
    knn_params[f'knn_k1_{metric_name}'] = {'n_neighbors': 1, 'metric': partial(spknn.distance, metric=metric_name)}

model_zoo = {name: KNeighborsClassifier(**model_params, algorithm='ball_tree', n_jobs=42) for
    name, model_params in knn_params.items()}


@chika.config
class Config:
    seed: int = 0
    dataset: str = 'retina'
    balanced: bool = False
    train_subsample_factor: float = 0.7
    test_subsample_factor: float = 0.7
    trials: int = 5
    out_folder: str = 'csv'


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    loader_fn = DATASET_NAME_LOADER_MAP[cfg.dataset]
    X_train, X_test, y_train, y_test, gr_train, gr_test = loader_fn(random_seed=cfg.seed)

    # spike train should be in the form {t1, t2, t3, ...}
    X_train = np.cumsum(X_train, axis=1)
    X_test = np.cumsum(X_test, axis=1)

    logging.info(
        f'Dataset shape after preprocessing: train {X_train.shape}, test {X_test.shape}'
    )
    logging.info(
        f'Mean target values: train {y_train.mean()}, test {y_test.mean()}'
    )

    models = {f'raw_{model_name}': model for model_name, model in model_zoo.items()}

    if cfg.balanced:
        results = subsampled_fit_predict(models, X_train, X_test, y_train, y_test, cfg)
        results.to_csv(f'{cfg.out_folder}/{cfg.dataset}_raw_spkknn_balanced.csv', index=False)
    else:
        results = {'model_name': [], 'gmean': [], 'cohen_kappa': []}
        for model_name, model in models.items():

            logging.info(f'Fitting model {model_name}')
            n = 1000
            idx = np.random.choice(range(X_train.shape[0]), n)
            model.fit(X_train[idx, :], y_train[idx])
            test_preds = model.predict(X_test)

            results['model_name'].append(model_name)
            results['gmean'].append(geometric_mean_score(y_test, test_preds))
            results['cohen_kappa'].append(cohen_kappa_score(y_test, test_preds))

            res_table = pd.DataFrame(results)
            logging.info(f'Validation metrics\n{res_table}')
            res_table.to_csv(f'{cfg.out_folder}/{cfg.dataset}_raw_spkknn_imbalanced.csv', index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
