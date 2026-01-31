import json
import os
import random
from datetime import datetime

import numpy as np


class EnvironmentManager:
    def __init__(self, base_dir="environments"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _generate_dir_path(self, params):
        """根据关键参数生成唯一目录名"""
        # 提取关键参数用于目录命名，方便识别
        N = params.get("EV_num", 0)
        K = params.get("BSS_num", 0)
        seed = params.get("rand_seed", "None")
        dist = params.get("distribution", "uniform")

        dir_name = f"N{N}_K{K}_{dist}_Seed{seed}"
        return os.path.join(self.base_dir, dir_name)

    def generate_environment(self, params):
        """根据参数字典生成环境数据"""
        EV_num = params["EV_num"]
        BSS_num = params["BSS_num"]
        area_width = params.get("area_width", 20)
        max_Battery_num = params.get("max_Battery_num", 15)
        rand_seed = params.get("rand_seed", None)
        distribution = params.get("distribution", "uniform")

        # 设置随机种子
        if rand_seed is not None:
            random.seed(rand_seed)
            np.random.seed(rand_seed)

        # 生成位置数据
        if distribution == 'gaussian':
            # 模拟热点区域：BSS均匀分布，EV围绕某个热点高斯分布
            BSS_location = np.random.uniform(0, area_width, (BSS_num, 2))
            center_bss = BSS_location[np.random.randint(0, BSS_num)]
            EV_location = np.random.normal(center_bss, 0.1 * area_width, (EV_num, 2))
            # 修正越界坐标
            EV_location = np.clip(EV_location, 0, area_width)
        else:
            # 默认均匀分布
            EV_location = np.random.uniform(0, area_width, (EV_num, 2))
            BSS_location = np.random.uniform(0, area_width, (BSS_num, 2))

        # 生成电池配置
        # 确保每个BSS至少有一定数量电池
        Battery_nums = np.random.randint(max(1, max_Battery_num // 3), max_Battery_num + 1, BSS_num)

        Inital_battery_Soc = [
            [random.uniform(0.5, 1) for _ in range(num)]
            for num in Battery_nums
        ]

        # 生成EV初始状态 (SoC_zero, SoC_hat, SoC_thr)
        EV_SoC = []
        for _ in range(EV_num):
            SoC_zero = random.uniform(0.2, 0.4)
            SoC_hat = random.uniform(SoC_zero / 3, SoC_zero * 2 / 3)
            SoC_thr = random.uniform(0.5, 0.7)
            EV_SoC.append([SoC_zero, SoC_hat, SoC_thr])

        # 计算距离矩阵 l_n_k
        l_n_k = np.zeros((EV_num, BSS_num))
        for i in range(EV_num):
            for j in range(BSS_num):
                l_n_k[i, j] = np.linalg.norm(EV_location[i] - BSS_location[j])

        # 清除种子影响
        random.seed(None)
        np.random.seed(None)

        return {
            "params": params,
            "data": {
                "EV_num": EV_num,
                "BSS_num": BSS_num,
                "EV_location": EV_location.tolist(),
                "BSS_location": BSS_location.tolist(),
                "Battery_nums": Battery_nums.tolist(),
                "Inital_battery_Soc": Inital_battery_Soc,
                "EV_SoC": np.array(EV_SoC).T.tolist(),
                "l_n_k": l_n_k.tolist()
            }
        }

    def save_environment(self, env_data):
        """保存环境到文件"""
        dir_path = self._generate_dir_path(env_data["params"])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"env_{timestamp}.json"
        file_path = os.path.join(dir_path, filename)

        with open(file_path, 'w') as f:
            json.dump(env_data, f, indent=4)

        print(f"[Info] Environment saved to: {file_path}")
        return file_path

    def load_environment(self, file_path):
        """载入环境"""
        with open(file_path, 'r') as f:
            content = json.load(f)
        return content["params"], content["data"]


# 使用示例
if __name__ == "__main__":
    manager = EnvironmentManager()
    # 生成一个测试环境
    params = {
        "EV_num": 20, "BSS_num": 5, "rand_seed": 42,
        "distribution": "uniform", "max_Battery_num": 10
    }
    env = manager.generate_environment(params)
    manager.save_environment(env)
