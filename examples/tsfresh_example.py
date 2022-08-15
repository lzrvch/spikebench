import logging
import sys
import warnings
from dataclasses import asdict

import chika
from imblearn.metrics import geometric_mean_score
from spikebench import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from spikebench.helpers import (set_random_seed, simple_undersampling,
                                tsfresh_vectorize)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, roc_auc_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}


@chika.config
class ModelConfig:
    n_estimators: int = 200
    max_depth: int = 10


@chika.config
class Config:
    model: ModelConfig
    seed: int = 0
    dataset: str = 'retina'
    balance_train: bool = False
    balance_test: bool = False
    tsfresh_scale_features: bool = True
    tsfresh_remove_low_variance: bool = True
    tsfresh_feature_set: str = 'distribution_features'
    window_size: int = 200
    step_size: int = 100


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    loader_fn = DATASET_NAME_LOADER_MAP[cfg.dataset]
    X_train, X_test, y_train, y_test, gr_train, gr_test = loader_fn(random_seed=cfg.seed)
    if cfg.balance_train:
        X_train, y_train = simple_undersampling(X_train, y_train)
    if cfg.balance_test:
        X_test, y_test = simple_undersampling(X_test, y_test)

    X_train, X_test, y_train, y_test = tsfresh_vectorize(X_train, X_test, y_train, y_test, cfg)

    logging.info(
        f'Dataset shape after preprocessing: train {X_train.shape}, test {X_test.shape}'
    )
    logging.info(
        f'Mean target values: train {y_train.mean()}, test {y_test.mean()}'
    )

    model = RandomForestClassifier(
        **asdict(cfg.model),
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    gmean = geometric_mean_score(y_test, model.predict(X_test))
    kappa = cohen_kappa_score(y_test, model.predict(X_test))
    logging.info(f'AUC ROC score value on {cfg.dataset} test set (random forest): {roc_auc}')
    logging.info(f'G-mean score value on {cfg.dataset} test set (random forest): {gmean}')
    logging.info(f'Kappa score value on {cfg.dataset} test set (random forest): {kappa}')


if __name__ == '__main__':
    main()
