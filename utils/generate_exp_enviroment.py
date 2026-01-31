import copy
import json
import os
import random
import sys
import numpy as np

from utils.parser_config import get_env_param


# 均匀随机生成
def generate_exp_enviroment_uniform(EV_num, BSS_num, area_width=20, max_Battery_num=15, rand_seed=None):
    # rng = random.Random()
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    EV_location = np.random.uniform(0, area_width, (EV_num, 2))
    BSS_location = np.random.uniform(0, area_width, (BSS_num, 2))
    Battery_nums = np.random.randint(max_Battery_num // 3, max_Battery_num, BSS_num)
    # Battery_nums = np.ones(BSS_num) * max_Battery_num
    Inital_battery_Soc = [[random.uniform(0.5, 1) for _ in range(Battery_nums[i])] for i in range(BSS_num)]
    EV_SoC = []
    for i in range(EV_num):
        SoC_zero = random.uniform(0.2, 0.4)
        SoC_hat = random.uniform(SoC_zero / 3, SoC_zero * 2 / 3)
        SoC_thr = random.uniform(0.5, 0.7)
        EV_SoC.append([SoC_zero, SoC_hat, SoC_thr])
    l_n_k = np.zeros((EV_num, BSS_num))
    for i in range(EV_num):
        for j in range(BSS_num):
            l_n_k[i, j] = np.linalg.norm(EV_location[i] - BSS_location[j])
    ret = {
        "EV_num": EV_num,
        "BSS_num": BSS_num,
        "EV_location": EV_location.tolist(),
        "BSS_location": BSS_location.tolist(),
        "Battery_nums": Battery_nums.tolist(),
        "Inital_battery_Soc": Inital_battery_Soc,
        "EV_SoC": np.array(EV_SoC).T.tolist(),
        "l_n_k": l_n_k.tolist()
    }
    random.seed(None)
    np.random.seed(None)
    return ret

# 所有EV聚集在第一个BSS附近，竞争资源
def generate_exp_enviroment_gaussian(EV_num, BSS_num, area_width=20, max_Battery_num=15, rand_seed=None):
    if rand_seed:
        np.random.seed(rand_seed)
        random.seed(rand_seed)
    BSS_location = np.random.uniform(0, area_width, (BSS_num, 2))
    EV_location = np.random.normal(BSS_location[0], 0.05 * area_width, (EV_num, 2))
    Battery_nums = np.random.randint(max_Battery_num // 3, max_Battery_num, BSS_num)
    # Battery_nums = np.ones(BSS_num) * max_Battery_num
    Inital_battery_Soc = [[random.uniform(0.5, 1) for _ in range(Battery_nums[i])] for i in range(BSS_num)]
    EV_SoC = []
    for i in range(EV_num):
        SoC_zero = random.uniform(0.2, 0.4)
        SoC_hat = random.uniform(SoC_zero / 3, SoC_zero * 2 / 3)
        SoC_thr = random.uniform(0.5, 0.7)
        EV_SoC.append([SoC_zero, SoC_hat, SoC_thr])
    l_n_k = np.zeros((EV_num, BSS_num))
    for i in range(EV_num):
        for j in range(BSS_num):
            l_n_k[i, j] = np.linalg.norm(EV_location[i] - BSS_location[j])
    ret = {
        "EV_num": EV_num,
        "BSS_num": BSS_num,
        "EV_location": EV_location.tolist(),
        "BSS_location": BSS_location.tolist(),
        "Battery_nums": Battery_nums.tolist(),
        "Inital_battery_Soc": Inital_battery_Soc,
        "EV_SoC": np.array(EV_SoC).T.tolist(),
        "l_n_k": l_n_k.tolist()
    }
    return ret

# 手动构造BSS与EV的位置关系
# undo
def generate_exp_enviroment_special(EV_num, BSS_num, area_width=20, max_Battery_num=15, rand_seed=None):
    if rand_seed:
        np.random.seed(rand_seed)
        random.seed(rand_seed)
    # EV_location = np.random.uniform(0, area_width, (EV_num, 2))
    # BSS_location = np.random.uniform(0, area_width, (BSS_num, 2))
    BSS_location = []
    EV_location = []
    Battery_nums = np.random.randint(max_Battery_num // 3, max_Battery_num, BSS_num)
    Inital_battery_Soc = [[random.uniform(0.5, 1) for _ in range(Battery_nums[i])] for i in range(BSS_num)]
    EV_SoC = []
    for i in range(EV_num):
        SoC_zero = random.uniform(0.2, 0.4)
        SoC_hat = random.uniform(SoC_zero / 3, SoC_zero * 2 / 3)
        SoC_thr = random.uniform(0.5, 0.7)
        EV_SoC.append([SoC_zero, SoC_hat, SoC_thr])
    l_n_k = np.zeros((EV_num, BSS_num))
    for i in range(EV_num):
        for j in range(BSS_num):
            l_n_k[i, j] = np.linalg.norm(EV_location[i] - BSS_location[j])
    ret = {
        "EV_num": EV_num,
        "BSS_num": BSS_num,
        "EV_location": EV_location.tolist(),
        "BSS_location": BSS_location.tolist(),
        "Battery_nums": Battery_nums.tolist(),
        "Inital_battery_Soc": Inital_battery_Soc,
        "EV_SoC": np.array(EV_SoC).T.tolist(),
        "l_n_k": l_n_k.tolist()
    }
    return ret

# 获取数据集
def get_exp_enviroment_dataset(env_param,dataset_name='dataset/uniform.json'):
    N = env_param["EV_num"]
    K = env_param["BSS_num"]
    rand_seed = env_param["random_seed"]
    area_width = env_param["area_width"]
    max_Battery_num = env_param["max_Battery_num"]
    # 判断数据集是否已经存在
    if not os.path.exists(dataset_name):
        dic=generate_exp_enviroment_uniform(N, K, area_width=area_width, max_Battery_num=max_Battery_num, rand_seed=rand_seed)
        json.dump(dic,open(dataset_name,'w'))
    dic=json.load(open(dataset_name,'r'))
    return  dic