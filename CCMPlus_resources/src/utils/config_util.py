import os
from argparse import Namespace

import yaml

EXP_DIR = os.path.join(os.path.dirname(__file__), "../../exps")
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_parser(config_path):
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return recursive_convert_dict(config)


def recursive_convert_dict(dic):
    for key, val in dic.items():
        if isinstance(val, dict):
            dic[key] = recursive_convert_dict(dic[key])
        else:
            continue
    return Namespace(**dic)
