from easydict import EasyDict as edict
import yaml


def parse_yaml_to_dict(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f, Loader=yaml.FullLoader))
    return parser
