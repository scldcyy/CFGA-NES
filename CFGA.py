import copy
import json
import random
import sys

import numpy as np
from tqdm import tqdm

# 移除对 objective 的全局引用
# from objective import ...
from utils.parser_config import get_GA_param


def init_pop(problem, pop_size=100):
    """
    初始化种群，确保生成的个体尽可能满足约束。
    依赖 problem 实例提供的环境参数。
    """
    pop = []
    # 尝试 pop_size * 100 次，以防止死循环
    for g in range(pop_size * 100):
        if len(pop) == pop_size:
            break

        # 使用 problem.bts
        bts_copy = copy.deepcopy(problem.bts)
        random.shuffle(bts_copy)
        sq = []

        for ev in range(problem.N):
            for bt in bts_copy:
                k, j = bt
                # 使用 problem 中的参数
                # Soc_nk 计算：当前电量 - 行驶耗电
                # 注意：problem.ln 是矩阵 [ev, k]
                Soc_nk = problem.E_init[ev] - problem.ln[ev, k] * problem.delta_yita_a / problem.En

                # 检查两个约束：
                # 1. 到达BSS时剩余电量 >= SoC_hats
                # 2. 目标电池初始电量 >= SoC_thrs
                # problem.E_battery_init 是列表结构 [k][j]
                if Soc_nk >= problem.SoC_hats[ev] and problem.E_battery_init[k][j] >= problem.SoC_thrs[ev]:
                    sq.append(bt)
                    bts_copy.remove(bt)
                    break
            else:
                # 如果某个EV找不到满足条件的电池，该个体生成失败
                # break内层循环，进入下一次尝试
                break

        # 只有当所有EV都找到了电池（len(sq) == N），才算成功
        if len(sq) == problem.N:
            pop.append(sq)

    if len(pop) != pop_size:
        # 如果尝试多次仍无法填满种群，抛出错误
        sys.exit(f"种群大小不足,大小为:{len(pop)}")
    return pop


def rank(pop, fitness):
    """
    对种群按适应度进行排序
    """
    pop_rank = np.argsort(-fitness)  # 降序，适应度越高越好
    rank_pop = [pop[r] for r in pop_rank]
    rank_fit = fitness[pop_rank]
    return rank_pop, rank_fit


def select(pop, fitness, eliteSize, pop_size):
    """
    精英选择策略 + 轮盘赌选择
    """
    fit_p = fitness / np.sum(fitness)
    # 保留前 eliteSize 个精英，剩余位置用轮盘赌填充
    select_pop = pop[:eliteSize] + random.choices(pop, weights=fit_p, k=pop_size - eliteSize)
    return select_pop


def cross(p1, p2, problem, crossRate=0.8):
    """
    OX (Order Crossover) 算子
    需要传入 problem 以获取 N
    """
    if random.random() < crossRate:
        # 随机选择交叉片段
        start, end = np.sort(np.random.choice(np.arange(problem.N), 2, False))

        # 生成子代 c1
        c1 = p1[start:end]
        # 遍历 p2（重新排序以保持相对顺序），将不在 c1 中的基因追加进去
        for gene in p2[end:] + p2[:end]:
            if gene not in c1:
                c1.append(gene)
            if len(c1) == problem.N:
                break
        # 恢复顺序
        c1 = c1[-start:] + c1[:-start]

        # 生成子代 c2
        c2 = p2[start:end]
        for gene in p1[end:] + p1[:end]:
            if gene not in c2:
                c2.append(gene)
            if len(c2) == problem.N:
                break
        c2 = c2[-start:] + c2[:-start]
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


def mutate(c, mutationRate, problem):
    """
    变异算子：包含替换变异和交换变异
    """
    # 替换变异：尝试从未使用过的电池集合中替换
    set_new_x = set(c)
    # problem.set_all 包含了所有可用电池组合
    replace = list(problem.set_all - set_new_x)
    random.shuffle(replace)
    d = 0
    for j in range(len(c)):
        if random.random() < mutationRate and d < len(replace):
            c[j] = replace[d]
            d += 1
        # 交换变异：交换两个EV的选择
        if random.random() < mutationRate:
            a = random.randint(0, problem.N - 1)
            c[a], c[j] = c[j], c[a]
    return c


def CFGA(problem, pop_size=500, maxFEs=500000, eliteSize=50, mutationRate=0.01, crossRate=0.8, obj=None):
    """
    CFGA 主流程
    参数:
        problem: BSSProblem 实例
        obj: 目标函数，默认使用 problem.objective
    """
    if obj is None:
        obj = problem.objective

    pbar = tqdm(total=maxFEs)
    GA_result = {
        "best_solution": [],
        "best_fit": [],
        "best_sum_fit": 0,
        "iter_fit": [],
        "iter_sum_fit": [],
    }

    # 初始化种群
    pop = init_pop(problem, pop_size=pop_size)

    # 计算初始适应度
    fitness = np.array([obj(x) for x in pop])
    FEs = pop_size

    while FEs < maxFEs:
        # 排序
        rank_pop, rank_fit = rank(pop, fitness)
        # 选择
        pop = select(rank_pop, rank_fit, eliteSize, pop_size)

        # 交叉变异（保留精英不参与，从 eliteSize 开始）
        for i in range(eliteSize, pop_size, 2):
            if i + 1 >= pop_size:
                break
            p1, p2 = pop[i], pop[i + 1]
            c1, c2 = cross(p1, p2, problem, crossRate)
            c1 = mutate(c1, mutationRate, problem)
            c2 = mutate(c2, mutationRate, problem)
            pop[i], pop[i + 1] = c1, c2

        # 重新计算适应度
        fitness = np.array([obj(x) for x in pop])
        FEs += pop_size

        # 记录最佳结果
        best_idx = np.argmax(fitness)
        GA_result["best_solution"] = pop[best_idx]

        # 使用 problem.f 获取详细的 cost 和 punnish 信息用于记录
        best_costs, _ = problem.f(pop[best_idx])

        GA_result["best_fit"] = best_costs.tolist()
        GA_result["best_sum_fit"] = sum(GA_result["best_fit"])
        GA_result["iter_fit"].append(GA_result["best_fit"])
        GA_result["iter_sum_fit"].append(GA_result["best_sum_fit"])

        pbar.set_postfix(bar_format=f"FEs:{FEs}, fit:{fitness[best_idx]:.5f}")
        pbar.update(pop_size)

    pbar.close()
    return GA_result


if __name__ == "__main__":
    # 为了保持脚本可运行性，这里需要手动实例化 Problem
    # 假设有一个默认的环境文件供测试
    from utils.generate_exp_enviroment import EnvironmentManager
    from objective import BSSProblem

    # 加载测试参数
    param = get_GA_param()
    random.seed(param["random_seed"])

    # 临时生成一个环境用于测试 main 函数
    env_manager = EnvironmentManager()
    # 假设使用小规模参数测试
    env_params = {"EV_num": 20, "BSS_num": 5, "rand_seed": param["random_seed"]}
    env_data = env_manager.generate_environment(env_params)
    problem = BSSProblem(env_data["data"])

    pop_size = param["pop_size"]
    maxFEs = param["maxFEs"]
    eliteSize = param["eliteSize"]
    mutationRate = param["mutationRate"]
    crossRate = param["crossRate"]

    GA_result = CFGA(problem, pop_size=pop_size, maxFEs=maxFEs, eliteSize=eliteSize,
                     mutationRate=mutationRate, crossRate=crossRate)

    json.dump(GA_result, open("GA_result_test.json", "w"), indent=4)
