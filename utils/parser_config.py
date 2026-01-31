import os

import yaml

param_path = 'param_small_scale.yml'
cur=os.getcwd()
with open(param_path, 'r', encoding='utf-8') as file:
    param = yaml.load(file, Loader=yaml.FullLoader)


def get_env_param():
    return param["enviroment_generate"]


def get_GA_param():
    return param["GA_param"]


def get_NES_param():
    return param["NES_param"]
