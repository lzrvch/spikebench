import argparse
import os

try:
    import jstyleson as json
except ImportError:
    import json

from addict import Dict


class ActionWrapper(argparse.Action):
    def __init__(self, action):
        self._action = action
        super().__init__(
            action.option_strings,
            action.dest,
            nargs=action.nargs,
            const=action.const,
            default=action.default,
            type=action.type,
            choices=action.choices,
            required=action.required,
            help=action.help,
            metavar=action.metavar,
        )
        self._action = action

    def __getattr__(self, item):
        return getattr(self._action, item)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.seen_actions.add(self._action.dest)
        return self._action(parser, namespace, values, option_string)


class CustomArgumentGroup(argparse._ArgumentGroup):
    def _add_action(self, action):
        super()._add_action(ActionWrapper(action))


class CustomActionContainer(argparse._ActionsContainer):
    def add_argument_group(self, *args, **kwargs):
        group = CustomArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group


class CustomArgumentParser(CustomActionContainer, argparse.ArgumentParser):
    """ArgumentParser that saves which arguments are provided"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.seen_actions = set()

    def parse_known_args(self, args=None, namespace=None):
        self.seen_actions.clear()
        return super().parse_known_args(args, namespace)


class Config(Dict):
    def __getattr__(self, item):
        if item not in self:
            raise KeyError('Key {} not found in config'.format(item))
        return super().__getattr__(item)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def update_from_args(self, args, argparser=None):
        if argparser is not None:
            if isinstance(argparser, CustomArgumentParser):
                default_args = {
                    arg for arg in vars(args) if arg not in argparser.seen_actions
                }
            else:
                # this will fail if we explicitly provide default argument in
                # CLI
                known_args = argparser.parse_known_args()
                default_args = {k for k, v in vars(args).items() if known_args[k] == v}
        else:
            default_args = {k for k, v in vars(args).items() if v is None}

        for key, value in vars(args).items():
            if key not in default_args or key not in self:
                self[key] = value

    def update_from_env(self, key_to_env_dict=None):
        for k, v in key_to_env_dict:
            if v in os.environ:
                self[k] = int(os.environ[v])


def get_common_argument_parser(default_config):
    parser = CustomArgumentParser()
    parser.add_argument('--window', default=default_config['window'], type=int)
    parser.add_argument('--step', default=default_config['step'], type=int)
    parser.add_argument('--trials', default=default_config['trials'], type=int)
    parser.add_argument('--seed', default=default_config['seed'], type=int)
    parser.add_argument(
        '--train-subsample-factor',
        default=default_config['train_subsample_factor'],
        type=float,
    )
    parser.add_argument(
        '--test-subsample-factor',
        default=default_config['test_subsample_factor'],
        type=float,
    )
    parser.add_argument(
        '--feature-set', default=default_config['feature_set'], type=str
    )
    parser.add_argument('--dataset', default=default_config['dataset'], type=str)
    parser.add_argument('--delimiter', default=default_config['delimiter'], type=str)
    parser.add_argument(
        '--remove_low_variance',
        default=default_config['remove_low_variance'],
        type=bool,
    )
    parser.add_argument('--scale', default=default_config['scale'], type=bool)
    return parser
