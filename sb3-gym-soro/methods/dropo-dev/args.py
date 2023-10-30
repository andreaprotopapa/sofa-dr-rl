"""Argument parser

    Priority of arguments:
        CLI parameters > <config-file>.yaml > default.yaml

    Examples:
        python script.py config=[conf1,conf2] param=value  # conf1.yaml must exist in CONFIG_PATH/
"""
from collections import Mapping
import os
from datetime import datetime
import pdb
import random
import string

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(f"Package omegaconf not installed.")

CONFIG_PATH = 'configs'
DEFAULT_CONFIG = 'default.yaml'
PARAMS_AS_LIST = []  # Params to interpret as lists

def add_extension(config_file):
    assert type(config_file) == str
    filename, _ = os.path.splitext(config_file)  # Returns filename and extension
    return filename+".yaml"

def pformat_dict(d, indent=0):
    fstr = ""
    for key, value in d.items():
        fstr += '\n' + '  ' * indent + str(key) + ":"
        if isinstance(value, Mapping):
            fstr += pformat_dict(value, indent+1)
        else:
            fstr += ' ' + str(value)
    return fstr

def to_dict(args):
    return OmegaConf.to_container(args)

def as_list(arg):
    if isinstance(arg, str):
        return [arg]
    elif isinstance(arg._content, list):
        return arg
    else:
        raise ValueError(f"This parameter was neither a string nor a list: {arg}")

def pars_as_list(args, keys):
    for key in keys:
        args[key] = as_list(args[key])
    return args

def save_config(config, path, filename='config.yaml'):
    with open(os.path.join(path, filename), 'w', encoding='utf-8') as file:
        OmegaConf.save(config=config, f=file.name)
    return

def create_dirs(path):
    try:
        os.makedirs(os.path.join(path))
    except OSError as error:
        pass

def get_random_string(n=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))



conf_path = os.path.join(os.path.dirname(__file__), CONFIG_PATH)  # Config path

args = OmegaConf.load(os.path.join(conf_path, DEFAULT_CONFIG))  # Default config file
cli_args = OmegaConf.from_cli()  # Read the cli args

if 'config' in cli_args and cli_args.config:  # Read configs (potentially as a list of configs)
    cli_args.config = [cli_args.config] if type(cli_args.config) == str else cli_args.config
    for i in range(len(cli_args.config)):
        cli_args.config[i] = add_extension(cli_args.config[i])  # Add .yaml extension if not present
        conf_args = OmegaConf.load(os.path.join(conf_path, cli_args.config[i]))
        args = OmegaConf.merge(args, conf_args)

args = OmegaConf.merge(args, cli_args)  # Merge cli args into config ones (potentially overwriting config args)
args = pars_as_list(args, PARAMS_AS_LIST)  # Convert parameters to lists if they are strings
# args = OmegaConf.to_object(args)  # Recursively convert your OmegaConf object to a plain python object (ListConfig to python list) -> args.dataset would no longer work, in favor of args['dataset']


### HARD CONSTRAINTS on args



### TODO
### save file as args.py
### in main:
###        from args import all

###        random_str = get_random_string(5)
###        # set_seed(args.seed)

###        run_name = random_str+('_'+args.name if args.name is not None else '')+'-S'+str(args.seed)
###        save_dir = os.path.join((args.output_dir if not args.debug else 'debug_runs'), run_name)

###        create_dirs(save_dir)
###        save_config(args, save_dir)

###        print('\n ===== RUN NAME:', run_name, f' ({save_dir}) ===== \n')
###        print(pformat_dict(args, indent=0))