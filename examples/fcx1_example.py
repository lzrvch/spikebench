import logging
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pyspikelib.config import get_common_argument_parser, Config
from pyspikelib import load_fcx1
from pyspikelib.helpers import tsfresh_fit_predict


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def get_argument_parser():
    default_config = {
        'seed': 0,
        'trials': 10,
        'scale': True,
        'remove_low_variance': True,
        'train_subsample_factor': 0.7,
        'test_subsample_factor': 0.7,
        'feature_set': 'distribution_features',
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

    X_train, X_test, y_train, y_test, gr_train, gr_test = load_fcx1(random_seed=config.seed)

    model = RandomForestClassifier(
        n_estimators=config.n_trees,
        random_state=config.seed,
        max_depth=10,
        n_jobs=-1
    )

    results = tsfresh_fit_predict(model, X_train, X_test, y_train, y_test, config)
    logging.info('Classification metrics:\n' + str(results.head(10)))


if __name__ == '__main__':
    main(sys.argv[1:])
