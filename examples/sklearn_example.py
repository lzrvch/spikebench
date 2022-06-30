import logging
import sys
import warnings

import chika
from dataclasses import asdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from pyspikelib import load_fcx1, load_retina, load_allen, load_fcx1_temporal
from pyspikelib.helpers import set_random_seed, simple_undersampling


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

    f1_value = f1_score(y_test, model.predict(X_test))
    logging.info(f'F1 score value on {cfg.dataset} test set (random forest): {f1_value}')


if __name__ == '__main__':
    main()
