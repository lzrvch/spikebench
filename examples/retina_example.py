import logging
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from examples.utils.config import get_common_argument_parser, Config
from examples.utils.dataset_adapters import retina_dataset
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
        'delimiter': None,
        'feature_set': None,
        'dataset': './data/retina/mode_paper_data',
        'n_trees': 200,
        'fstate': 'randomly_moving_bar',
        'mstate': 'white_noise_checkerboard',
    }
    parser = get_common_argument_parser(default_config)
    parser.add_argument('--n-trees', default=default_config['n_trees'], type=int)
    parser.add_argument('--fstate', default=default_config['fstate'], type=str)
    parser.add_argument('--mstate', default=default_config['mstate'], type=str)
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = Config()
    config.update_from_args(args, parser)

    np.random.seed(config.seed)
    retinal_spike_data = retina_dataset(config.dataset)

    group_split = GroupShuffleSplit(n_splits=1, test_size=0.5)
    X = np.hstack(
        [
            retinal_spike_data[config.fstate].series.values,
            retinal_spike_data[config.mstate].series.values,
        ]
    )
    y = np.hstack(
        [
            np.ones(retinal_spike_data[config.fstate].shape[0]),
            np.zeros(retinal_spike_data[config.mstate].shape[0]),
        ]
    )
    groups = np.hstack(
        [
            retinal_spike_data[config.fstate].groups.values,
            retinal_spike_data[config.mstate].groups.values,
        ]
    )

    for train_index, test_index in group_split.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = pd.DataFrame({'series': X_train, 'groups': groups[train_index]})
    X_test = pd.DataFrame({'series': X_test, 'groups': groups[test_index]})

    forest = RandomForestClassifier(
        n_estimators=config.n_trees,
        random_state=config.seed,
        max_depth=10,
        n_jobs=-1
    )
    results = tsfresh_fit_predict(forest, X_train, X_test, y_train, y_test, config)
    logging.info('Classification metrics:\n' + str(results.head(10)))


if __name__ == '__main__':
    main(sys.argv[1:])
