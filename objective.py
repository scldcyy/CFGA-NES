# 用于生成环境与目标函数

import copy

import numpy as np

from utils.generate_exp_enviroment import get_exp_enviroment_dataset
from utils.parser_config import get_env_param

env_param = get_env_param()
N = env_param["EV_num"]
K = env_param["BSS_num"]
alpha = 0.5
beta = 0.5
tao = 0.6
dic=get_exp_enviroment_dataset(env_param)
E_init = np.array(dic["EV_SoC"][0])
SoC_hats = np.array(dic["EV_SoC"][1])
SoC_thrs = np.array(dic["EV_SoC"][2])
Jk = np.array(dic["Battery_nums"])
# print("总电池数：", sum(Jk))
batterys = []
for i in range(len(Jk)):
    for j in range(Jk[i]):
        batterys.append((i, j))
sa = 50
delta_yita_a = 7.344 * 1e-5 * sa ** 2 - 7.656 * 1e-3 * sa + 0.3536
E_battery_init = dic["Inital_battery_Soc"]
ln = np.array(dic["l_n_k"])
En = 75
Lk_max = Jk * En
#Lk_max=np.array([525,525,525,525,525])
Rk = 12.5 * Jk  # rho * delta_t=1.25*10
p_grid = 0.85
Lkavi_pre = np.array([np.sum(E_battery_init[k] * En) for k in range(K)])
bts = []
for i in range(len(Jk)):
    for j in range(Jk[i]):
        bts.append((i, j))
set_all = set(bts)


def f(x):  # 目标函数
    Lkavi_cur = Lkavi_pre + Rk
    Lk = np.zeros(K)
    enk = np.zeros(N)
    punnishs = np.zeros(N)
    for n in range(N):
        # 到车站约束 E_res>=SoC_hats
        k, j = x[n]
        E_res = E_init[n] - ln[n, k] * delta_yita_a / En
        # 到目的地约束 E_battery_init[x[i]]>=SoC_thrs
        en = (E_battery_init[k][j] - E_res) * En
        Lk[k] += en
        enk[n] = en
        # punnishs[n] = (max(0, SoC_hats[n] - E_res) + max(0, SoC_thrs[n] - E_battery_init[k][j]))*1e4
        punnishs[n] = 0 if E_res >= SoC_hats[n] and E_battery_init[k][j] > SoC_thrs[n] else 1e4
    Gammak = 2 * p_grid - Lkavi_cur / Lk_max
    Lambdak = 1 / Lk_max
    pk = Gammak + Lambdak * Lk
    ret = []
    for n in range(N):
        k, j = x[n]
        tmp = alpha * enk[n] * pk[k] + beta * tao * ln[n, k]
        ret.append(tmp)
    return np.array(ret), punnishs


def objective(x):
    return 1 / np.sum(f(x))

def potential_function(x):
    Lkavi_cur = Lkavi_pre + Rk
    Lk = np.zeros(K)
    Qk = np.zeros(K)
    enk = np.zeros(N)
    punnishs = np.zeros(N)
    for n in range(N):
        # 到车站约束 E_res>=SoC_hats
        k, j = x[n]
        E_res = E_init[n] - ln[n, k] * delta_yita_a / En
        # 到目的地约束 E_battery_init[x[i]]>=SoC_thrs
        en = (E_battery_init[k][j] - E_res) * En
        Lk[k] += en
        Qk[k] += en* en
        enk[n] = en
        # punnishs[n] = (max(0, SoC_hats[n] - E_res) + max(0, SoC_thrs[n] - E_battery_init[k][j]))*1e4
        punnishs[n] = 0 if E_res >= SoC_hats[n] and E_battery_init[k][j] > SoC_thrs[n] else 1e4
    Gammak=2*p_grid-Lkavi_cur/Lk_max
    Lambdak = 1/Lk_max
    pk=Gammak+Lambdak*Lk
    ret = []
    potential=0
    for n in range(N):
        k, j = x[n]
        tmp = alpha * enk[n] * pk[k] + beta * tao * ln[n, k]
        potential+=alpha* (Gammak[k]*Lk[k]+0.5*Lambdak[k]*(Lk[k]*Lk[k]+Qk[k]))+beta*tao*ln[n,k]
        ret.append(tmp)
    return np.array(ret), punnishs,potential


# 最佳响应总数量
def total_best_response(x):
    ret = len(x)
    cost, punnish = f(x)
    cand_set = list(set_all - set(x))
    print(f"### 计算最佳响应数量 ,x={x} ,cost={cost}, punnish={punnish}###")
    for i in range(len(x)):
        x_cp = copy.deepcopy(x)
        for s in cand_set:
            x_cp[i] = s
            new_cost, new_punnish = f(x_cp)
            # 无最佳响应
            if (new_cost[i] < cost[i] and new_punnish[i] == 0) or punnish[i] > 0:
                ret -= 1
                break
    return ret

# 计算ev_num号EV的最佳响应策略
def best_response_strategy(ev_num, x):
    x_cp = copy.deepcopy(x)
    cand_set = list(set_all - set(x_cp))
    ret = x_cp[ev_num]
    min_cost, punnish = f(x)
    for s in cand_set:
        x_cp[ev_num] = s
        new_cost, new_punnish = f(x_cp)
        # 有新的最佳响应
        if new_cost[ev_num] < min_cost[ev_num] and new_punnish[ev_num] == 0:
            min_cost[ev_num] = new_cost[ev_num]
            ret = s
    return ret


if __name__ == "__main__":
    pass
