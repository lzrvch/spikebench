import logging
import sys
import warnings
from dataclasses import asdict

import chika
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from spikebench import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from spikebench.helpers import set_random_seed, tsfresh_vectorize

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}


@chika.config
class Config:
    seed: int = 0
    preprocessing: bool = True
    trials: int = 5
    tsfresh_feature_set: str = 'full'
    tsfresh_remove_low_variance: bool = True
    tsfresh_scale_features: bool = True
    top_k: int = 50


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    importance_scores = {}
    for dataset in ('retina', 'allen', 'fcx1'):
        cfg.dataset = dataset
        loader_fn = DATASET_NAME_LOADER_MAP[cfg.dataset]
        X_train, X_test, y_train, y_test, gr_train, gr_test = loader_fn(random_seed=cfg.seed)

        if cfg.preprocessing:
            scaler = StandardScaler()
            X_train = np.log1p(X_train)
            X_test = np.log1p(X_test)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        X_train, X_test, y_train, y_test = tsfresh_vectorize(X_train, X_test, y_train, y_test, cfg,
                cache_file=f'./bin/tsfresh_features_{cfg.dataset}_{cfg.tsfresh_feature_set}.bin')

        logging.info(
            f'Dataset shape after preprocessing: train {X_train.shape}, test {X_test.shape}'
        )
        logging.info(
            f'Mean target values: train {y_train.mean()}, test {y_test.mean()}'
        )

        model = RandomForestClassifier(
                n_estimators=500,
                random_state=0,
                max_depth=10,
                n_jobs=-1
            )
        model.fit(X_train, y_train)

        for index, feature in enumerate(X_train.columns):
            if feature not in importance_scores:
                importance_scores[feature] = []
            importance_scores[feature].append(model.feature_importances_[index])

    final_score = {}
    for feature in importance_scores:
        if len(importance_scores[feature]) == 3:
            final_score[feature] = np.mean(importance_scores[feature])

    df = pd.DataFrame({'feature': final_score.keys(), 'score': final_score.values()})
    logging.info('Features with top mean importance scores across datasets')
    logging.info(df.sort_values(by='score', ascending=False).feature.values[:cfg.top_k])


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
