import logging
import pickle
import sys
import warnings
from dataclasses import asdict

import chika
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from spikebench import load_allen, load_fcx1, load_fcx1_temporal, load_retina
from spikebench.helpers import set_random_seed
from xgboost import XGBClassifier

DATASET_NAME_LOADER_MAP = {
    'fcx1': load_fcx1,
    'retina': load_retina,
    'allen': load_allen,
    'fcx1_temporal': load_fcx1_temporal,
}

model_name = 'xgboost'


@chika.config
class Config:
    seed: int = 0
    dataset: str = 'retina'
    preprocessing: bool = True
    tsfresh_feature_set: str = None
    tsfresh_remove_low_variance: bool = True
    tsfresh_scale_features: bool = True
    xgb_early_stopping_rounds: int = 10
    hyperopt_steps: int = 100
    final_num_estimators: int = 500


@chika.main(cfg_cls=Config)
def main(cfg: Config):
    set_random_seed(cfg.seed)

    logging.info(
        f'Running job with config {asdict(cfg)}'
    )

    feature_sets = ('full', )
    for feature_set in feature_sets:
        cfg.tsfresh_feature_set = feature_set

        isi_cache_file = f'./bin/tsfresh_features_{cfg.dataset}_{cfg.tsfresh_feature_set}.bin'
        with open(isi_cache_file, 'rb') as f:
            X_train_isi, y_train, X_test_isi, y_test = pickle.load(f)

        sce_cache_file = f'./bin/tsfresh_features_{cfg.dataset}_{cfg.tsfresh_feature_set}_sce.bin'
        with open(sce_cache_file, 'rb') as f:
            X_train_sce, y_train, X_test_sce, y_test = pickle.load(f)

        X_train_sce.columns = np.array([feature_name + '_sce' for feature_name
            in X_train_sce.columns])
        X_test_sce.columns = np.array([feature_name + '_sce' for feature_name
            in X_test_sce.columns])

        X_train = pd.concat([X_train_isi, X_train_sce], axis=1)
        X_test = pd.concat([X_test_isi, X_test_sce], axis=1)

        logging.info(
            f'Dataset shape after preprocessing: train {X_train.shape}, test {X_test.shape}'
        )
        logging.info(
            f'Mean target values: train {y_train.mean()}, test {y_test.mean()}'
        )

        results = {'model_name': [], 'roc_auc': [], 'gmean': [], 'cohen_kappa': []}

        logging.info(f'Fitting model {model_name}')

        space = {
            'max_depth': hp.quniform('max_depth', 3, 18, 1),
            # 'gamma': 0,
            # 'reg_alpha': 0,
            # 'reg_lambda': 1,
            # 'colsample_bytree': 1,
            # 'min_child_weight': 1,
            'gamma': hp.uniform('gamma', 0, 3),
            'reg_alpha' : hp.quniform('reg_alpha', 0, 100, 1),
            'reg_lambda' : hp.uniform('reg_lambda', 0, 1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
            'n_estimators': 100,
            'seed': 0,
            'learning_rate': 1e-1,
            'verbosity': 0,
            'silent': None,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'n_jobs': -1,
            'nthread': None,
            'max_delta_step': 0,
            'subsample': 0.7,
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'scale_pos_weight': 1,
            'base_score': 0.5,
            'random_state': 0,
            'seed': 0,
        }

        def objective(space):
            space['max_depth'] = int(space['max_depth'])
            space['reg_alpha'] = int(space['reg_alpha'])
            space['min_child_weight'] = int(space['min_child_weight'])
            space['colsample_bytree'] = int(space['colsample_bytree'])

            model = XGBClassifier(
                **space
            )

            evaluation = [(X_train, y_train), (X_test, y_test)]

            model.fit(X_train, y_train,
                eval_set=evaluation, eval_metric='auc',
                early_stopping_rounds=cfg.xgb_early_stopping_rounds, verbose=False)

            test_preds = model.predict(X_test)
            score = cohen_kappa_score(y_test, test_preds)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()

        best_hyperparams = fmin(fn=objective,
                                space=space,
                                algo=tpe.suggest,
                                max_evals=cfg.hyperopt_steps,
                                trials=trials)

        space.update(best_hyperparams)
        space['n_estimators'] = cfg.final_num_estimators
        model = XGBClassifier(
            **space
        )

        model.fit(X_train, y_train)

        test_preds = model.predict(X_test)
        test_probas = model.predict_proba(X_test)[:, 1]

        results['model_name'].append(model_name)
        results['roc_auc'].append(roc_auc_score(y_test, test_probas))
        results['gmean'].append(geometric_mean_score(y_test, test_preds))
        results['cohen_kappa'].append(cohen_kappa_score(y_test, test_preds))

        res_table = pd.DataFrame(results)
        logging.info(f'Validation metrics\n{res_table}')
        res_table.to_csv(f'./csv/{cfg.dataset}_{feature_set}_sce_plus_isi_tsfresh_hyperopt_imbalanced.csv', index=False)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
