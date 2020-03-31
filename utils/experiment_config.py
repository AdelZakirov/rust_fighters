import argparse
import os
from pathlib import Path
import yaml


def validate_keys(conf):
    if isinstance(conf, dict):
        for k, v in conf.items():
            if '.' in k:
                raise ValueError(f'config key should not contain . ; violating key: {k}')
            validate_keys(v)


def read_experiment_config(path):
    cp = Path(path)
    if not cp.exists():
        raise FileNotFoundError(f'specified config path {path} does not exist')
    with open(path) as f:
        config = yaml.load(f)
    validate_keys(config)
    return argparse.Namespace(**config)


def experiment_config_cmdline(default=None):
    """Read experiment config from YAML file"""
    if 'CONFIG' in os.environ:
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(os.environ['CONFIG'])))
    else:
        path = default
    print(f"reading experiment config from {path}")
    return read_experiment_config(path)


if __name__ == '__main__':
    print(experiment_config_cmdline())
