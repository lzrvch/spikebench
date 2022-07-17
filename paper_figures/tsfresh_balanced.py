import logging
import sys
import warnings
from dataclasses import asdict

import chika
import pandas as pd
from pyspikelib import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from pyspikelib.helpers import (set_random_seed, subsampled_fit_predict,
                                tsfresh_vectorize)
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}


model_zoo = {
        'random_forest': RandomForestClassifier(
            n_estimators=500,
            random_state=0,
            max_depth=13,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=500,
            random_state=0,
            max_depth=None,
            n_jobs=-1
        ),
        'logit_l2': LogisticRegression(
            random_state=0,
            penalty='l2',
            C=0.01,
            n_jobs=-1,
        ),
        'logit': LogisticRegression(
            random_state=0,
            penalty='none',
            n_jobs=-1,
        ),
        'xgboost': XGBClassifier(
            max_depth=8,
            learning_rate=1e-1,
            n_estimators=1000,
            verbosity=0,
            silent=None,
            objective='binary:logistic',
            booster='gbtree',
            n_jobs=-1,
            nthread=None,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=0.7,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            random_state=0,
            seed=0
        )
}


@chika.config
class Config:
    seed: int = 0
    dataset: str = 'retina'
    train_subsample_factor: float = 0.7
    test_subsample_factor: float = 0.7
    trials: int = 10
    tsfresh_feature_set: str = None
    tsfresh_remove_low_variance: bool = True
    tsfresh_scale_features: bool = True


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    tsfresh_results = pd.DataFrame()
    feature_sets = ('simple_baseline', 'full')
    for feature_set in feature_sets:
        cfg.tsfresh_feature_set = feature_set
        loader_fn = DATASET_NAME_LOADER_MAP[cfg.dataset]
        X_train, X_test, y_train, y_test, gr_train, gr_test = loader_fn(random_seed=cfg.seed)

        X_train, X_test, y_train, y_test = tsfresh_vectorize(X_train, X_test, y_train, y_test, cfg,
            cache_file=f'./bin/tsfresh_features_{cfg.dataset}_{cfg.tsfresh_feature_set}.bin')

        logging.info(
            f'Dataset shape after preprocessing: train {X_train.shape}, test {X_test.shape}'
        )
        logging.info(
            f'Mean target values: train {y_train.mean()}, test {y_test.mean()}'
        )

        for model_name, model in model_zoo.items():
            results = subsampled_fit_predict(model, X_train, X_test, y_train, y_test, cfg)
            results['model'] = [f'tsfresh_{feature_set}_{model_name}'] * cfg.trials
            tsfresh_results = pd.concat([tsfresh_results, results], axis=0)

    tsfresh_results.to_csv(f'./csv/{cfg.dataset}_tsfresh_balanced.csv')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
