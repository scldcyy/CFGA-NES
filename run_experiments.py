import matplotlib.pyplot as plt
import numpy as np

# 假设 CFGA 和 NES 文件在同一目录下，并且我们需要对其做微小调整以接受 problem 对象
# 为方便运行，这里我将直接调用修改后的逻辑，或者你需要修改 CFGA.py/NES.py
# 让它们接受 obj_func 作为参数。
# 下面代码假设你已经修改 CFGA.py 的入口为 CFGA(..., problem=None)
import CFGA
import NES
from objective import BSSProblem
# 引入我们重构的模块
from utils.generate_exp_enviroment import EnvironmentManager


def run_greedy(problem):
    """
    GS (Greedy Strategy): 每个 EV 选择距离最近的 BSS。
    不保证满足冲突约束，如果冲突，随机选电池或按顺序选。
    """
    strategy = []
    # 记录每个BSS已被占用的电池索引
    occupied = {k: set() for k in range(problem.K)}

    for n in range(problem.N):
        # 找到最近的 BSS
        dists = problem.ln[n]
        # 按距离排序的 BSS 索引
        sorted_k = np.argsort(dists)

        chosen = False
        for k in sorted_k:
            # 在该 BSS 中找一个空闲电池
            available_batteries = list(set(range(problem.Jk[k])) - occupied[k])
            if available_batteries:
                # 简单贪心：选第一个可用电池（通常是索引最小的）
                j = available_batteries[0]
                strategy.append((k, j))
                occupied[k].add(j)
                chosen = True
                break

        if not chosen:
            # 如果所有站都满了（极端情况），随机分配一个防止报错，但会产生极大惩罚
            strategy.append((sorted_k[0], 0))

    return strategy


def run_experiment_pipeline():
    # 1. 准备实验环境
    manager = EnvironmentManager()

    # 定义实验参数（参考论文 Small Scale）
    params_small = {
        "EV_num": 20,
        "BSS_num": 5,
        "rand_seed": 3407,
        "distribution": "uniform",
        "max_Battery_num": 10
    }

    # 生成并保存环境
    env_path = manager.save_environment(manager.generate_environment(params_small))
    print(f"Loading environment from: {env_path}")
    _, env_data = manager.load_environment(env_path)

    # 初始化问题实例
    problem = BSSProblem(env_data)

    # 2. 定义实验配置
    REPEAT = 5  # 重复实验次数，论文通常为 20
    results = {
        "GS": [],
        "CFGA": [],
        "NES": [],
        "CFGA-NES": []
    }

    print("Starting Experiments...")

    for r in range(REPEAT):
        print(f"\n--- Round {r + 1}/{REPEAT} ---")

        # --- Run GS ---
        x_gs = run_greedy(problem)
        cost_gs, pun_gs = problem.f(x_gs)
        total_gs = np.sum(cost_gs) + np.sum(pun_gs)
        results["GS"].append(total_gs)

        # --- Run CFGA ---
        # 注意：你需要修改 CFGA.py 使其接受 problem 实例
        # 假设修改后的调用方式如下：
        # 这里我们在脚本里临时 Monkey Patch 一下 objective，或者传入参数
        # 为了演示，直接传入 problem.objective
        ga_res = CFGA.CFGA(pop_size=50, maxFEs=5000, obj=problem.objective)
        # CFGA.py 返回的是 best_solution
        x_cfga = ga_res["best_solution"]
        cost_cfga, pun_cfga = problem.f(x_cfga)
        results["CFGA"].append(np.sum(cost_cfga) + np.sum(pun_cfga))

        # --- Run NES (Random Init) ---
        # NES 需要初始解，这里随机生成一个无冲突解作为起点
        init_rand = CFGA.init_pop(1)[0]  # 复用 CFGA 的初始化
        # NES.py 也需要修改以接受 problem.f
        # 假设 NE_Seeking(..., fit_func=problem.f)
        # 暂时用全局变量替换法模拟（不推荐但在不修改原文件大量代码时有效）
        NES.f = problem.f
        NES.N = problem.N
        NES.set_all = problem.set_all

        nes_res = NES.NE_Seeking(I=50, init_bts=init_rand)
        results["NES"].append(nes_res["best_sum_fit"])

        # --- Run CFGA-NES ---
        # 使用 CFGA 的结果作为 NES 的起点
        one_res = NES.NE_Seeking(I=50, init_bts=x_cfga)
        results["CFGA-NES"].append(one_res["best_sum_fit"])

    # 3. 结果分析与打印
    print("\n=== Experiment Results (Total Cost) ===")
    for alg, data in results.items():
        mean_val = np.mean(data)
        std_val = np.std(data)
        print(f"{alg}: {mean_val:.2f} ± {std_val:.2f}")

    # 简单的可视化保存
    plt.figure()
    plt.boxplot(results.values(), labels=results.keys())
    plt.ylabel("Total Cost")
    plt.title("Algorithm Comparison")
    plt.savefig("experiment_comparison.png")
    print("Plot saved to experiment_comparison.png")


if __name__ == "__main__":
    run_experiment_pipeline()
