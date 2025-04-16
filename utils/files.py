import json 
import yaml
try:
    # Fast yaml (if available)
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def load_yaml(filename):
    """ Loads a YAML file in a Python dictionary. """
    with open(filename,'r') as json_file:
        data = yaml.load(json_file, Loader=Loader)
    return data


def load_json(filename):
    """ Loads a JSON file into a Python data structure. """
    with open(filename,'r') as json_file:
        data = json.load(json_file)
    return data