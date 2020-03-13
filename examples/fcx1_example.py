import sys
import logging
import warnings

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from examples.dataset_adapters import fcx1_dataset
from examples.config import get_common_argument_parser, Config
from pyspikelib.fit_predict import tsfresh_fit_predict

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def get_argument_parser():
    default_config = {
        'seed': 0,
        'window': 200,
        'step': 200,
        'trials': 10,
        'scale': True,
        'remove_low_variance': True,
        'train_subsample_factor': 0.7,
        'test_subsample_factor': 0.7,
        'delimiter': ',',
        'feature_set': None,
        'dataset': './data/fcx1',
        'n_trees': 200,
    }
    parser = get_common_argument_parser(default_config)
    parser.add_argument('--n-trees', default=default_config['n_trees'], type=int)
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = Config()
    config.update_from_args(args, parser)

    np.random.seed(config.seed)

    dataset_path = Path(config.dataset)
    wake_spikes = fcx1_dataset(dataset_path / 'wake.parq')
    sleep_spikes = fcx1_dataset(dataset_path / 'sleep.parq')

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.7)
    X = np.hstack([wake_spikes.series.values, sleep_spikes.series.values])
    y = np.hstack([np.ones(wake_spikes.shape[0]), np.zeros(sleep_spikes.shape[0])])
    groups = np.hstack([wake_spikes.groups.values, sleep_spikes.groups.values])

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    forest = RandomForestClassifier(
        n_estimators=config.n_trees, random_state=config.seed, n_jobs=-1
    )
    scores = tsfresh_fit_predict(forest, X_train, X_test, y_train, y_test, config)
    return scores


if __name__ == '__main__':
    main(sys.argv[1:])
