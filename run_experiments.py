import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import CFGA
import NES
from objective import BSSProblem
# 引入重构后的模块
from utils.generate_exp_enviroment import EnvironmentManager


class ExperimentRunner:
    def __init__(self, output_dir="experiment_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_greedy(self, problem):
        """
        GS (Greedy Strategy) 实现
        策略：每个EV贪心地选择距离最近且有可用电池的BSS。
        """
        strategy = [None] * problem.N
        # 记录每个BSS已被占用的电池索引集合，避免冲突
        occupied = {k: set() for k in range(problem.K)}

        # 简单的贪心：按EV顺序，或者按EV到最近站点的距离排序处理
        # 这里模拟独立决策，按顺序处理（更符合非合作博弈的无序性）
        for n in range(problem.N):
            # 获取该EV到所有BSS的距离
            dists = problem.ln[n]
            # 按距离从小到大排序的BSS索引
            sorted_k = np.argsort(dists)

            chosen = False
            for k in sorted_k:
                # 检查该BSS是否有空闲电池
                # problem.Jk[k] 是该站电池总数
                all_batteries = set(range(problem.Jk[k]))
                available = list(all_batteries - occupied[k])

                if available:
                    # 贪心选择：选第一个可用的（通常索引最小）
                    j = available[0]

                    # 还需要检查物理约束：到达电量和电池初始电量
                    # 这一步在简单贪心中有时会忽略，但在对比实验中最好加上，
                    # 否则GS会有巨大的惩罚成本，对比失去意义。
                    # 这里我们做一个“聪明的贪心”，只选满足约束的。

                    # 预计算约束条件
                    E_res = problem.E_init[n] - problem.ln[n, k] * problem.delta_yita_a / problem.En
                    E_bat = problem.E_battery_init[k][j]

                    if E_res >= problem.SoC_hats[n] and E_bat >= problem.SoC_thrs[n]:
                        strategy[n] = (k, j)
                        occupied[k].add(j)
                        chosen = True
                        break

            if not chosen:
                # 如果所有站都满了或不满足约束，被迫选择最近的站的第0个电池（由于冲突，这将导致巨大惩罚）
                # 这是为了保证策略完整性
                strategy[n] = (sorted_k[0], 0)

        return strategy

    def calculate_utilization(self, problem, strategy):
        """
        计算 BSS 利用率
        Utilization = (被交换的电池总能量 / BSS总容量)
        或者简单按 (被使用的电池数量 / BSS电池总数)
        论文中通常指电池使用数量比例或能量负载比例。这里使用负载比例 (Lk / Lk_max)。
        """
        costs, _ = problem.f(strategy)  # 触发内部计算，虽然这里不需要costs
        # 我们需要重新计算 Lk
        # 复用 problem.f 中的逻辑片段
        x_arr = np.array(strategy)
        ks = x_arr[:, 0]
        js = x_arr[:, 1]

        dists = problem.ln[np.arange(problem.N), ks]
        E_res = problem.E_init - dists * problem.delta_yita_a / problem.En
        E_bat_init_vals = np.array([problem.E_battery_init[k][j] for k, j in strategy])
        en = (E_bat_init_vals - E_res) * problem.En

        Lk = np.zeros(problem.K)
        np.add.at(Lk, ks, en)

        # 利用率 = Lk / Lk_max
        utilization = Lk / problem.Lk_max
        return utilization

    def run_comprehensive_benchmark(self, env_params, repeat_times=20):
        """
        运行完整的对比实验
        """
        # 1. 环境准备
        manager = EnvironmentManager()
        # 每次大实验生成一个固定的环境，确保所有算法在同一环境下竞争
        # 如果希望每次repeat都变环境，把这段移到循环里（通常benchmark是固定环境跑多次算法随机性，或跑多个环境取平均）
        # 这里采用：固定环境参数，每次repeat生成新的随机种子环境，以测试鲁棒性。

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(self.output_dir, f"Exp_{timestamp}")
        os.makedirs(exp_dir)

        # 存储所有数据的列表
        records = []

        print(f"Starting Benchmark: {repeat_times} rounds...")

        for r in tqdm(range(repeat_times), desc="Experiment Progress"):
            # 每一轮使用不同的随机种子生成环境，模拟不同的天/场景
            current_params = env_params.copy()
            current_params["rand_seed"] = r * 100 + 42  # 确保可复现

            env_data_wrapper = manager.generate_environment(current_params)
            problem = BSSProblem(env_data_wrapper["data"])

            # === Algorithm 1: Greedy (GS) ===
            start_time = time.time()
            x_gs = self.run_greedy(problem)
            gs_time = time.time() - start_time

            c_gs, p_gs = problem.f(x_gs)
            cost_gs = np.sum(c_gs) + np.sum(p_gs)
            util_gs = self.calculate_utilization(problem, x_gs)

            records.append({
                "Round": r, "Algorithm": "GS", "TotalCost": cost_gs,
                "Time": gs_time, "Utilization": util_gs.tolist(),
                "BestResponseCount": problem.total_best_response(x_gs)
            })

            # === Algorithm 2: CFGA ===
            start_time = time.time()
            # 传入 problem
            ga_res = CFGA.CFGA(problem, pop_size=50, maxFEs=5000,
                               eliteSize=5, mutationRate=0.01)
            cfga_time = time.time() - start_time

            x_cfga = ga_res["best_solution"]
            c_cfga, p_cfga = problem.f(x_cfga)
            cost_cfga = np.sum(c_cfga) + np.sum(p_cfga)
            util_cfga = self.calculate_utilization(problem, x_cfga)

            records.append({
                "Round": r, "Algorithm": "CFGA", "TotalCost": cost_cfga,
                "Time": cfga_time, "Utilization": util_cfga.tolist(),
                "BestResponseCount": problem.total_best_response(x_cfga)
            })

            # === Algorithm 3: NES (Random Init) ===
            start_time = time.time()
            # NES 内部会自己 init_pop(problem)
            nes_res = NES.NE_Seeking(problem, I=50, init_bts=None)
            nes_time = time.time() - start_time

            x_nes = nes_res["best_solution"]
            # cost 已在 res 中，但为了统一计算逻辑，再算一次
            c_nes, p_nes = problem.f(x_nes)
            cost_nes = np.sum(c_nes) + np.sum(p_nes)
            util_nes = self.calculate_utilization(problem, x_nes)

            records.append({
                "Round": r, "Algorithm": "NES", "TotalCost": cost_nes,
                "Time": nes_time, "Utilization": util_nes.tolist(),
                "BestResponseCount": problem.total_best_response(x_nes)
            })

            # === Algorithm 4: CFGA-NES (Proposed) ===
            start_time = time.time()
            # 使用 CFGA 的结果作为初始解
            one_res = NES.NE_Seeking(problem, I=50, init_bts=x_cfga)
            one_time = time.time() - start_time + cfga_time  # 累加 CFGA 的时间

            x_one = one_res["best_solution"]
            c_one, p_one = problem.f(x_one)
            cost_one = np.sum(c_one) + np.sum(p_one)
            util_one = self.calculate_utilization(problem, x_one)

            records.append({
                "Round": r, "Algorithm": "CFGA-NES", "TotalCost": cost_one,
                "Time": one_time, "Utilization": util_one.tolist(),
                "BestResponseCount": problem.total_best_response(x_one)
            })

        # === 保存结果 ===
        self.save_and_plot(records, exp_dir)

    def save_and_plot(self, records, exp_dir):
        # 1. 保存原始 JSON
        json_path = os.path.join(exp_dir, "raw_data.json")
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=4)
        print(f"Raw data saved to {json_path}")

        # 2. 转换为 DataFrame 方便分析
        df = pd.DataFrame(records)
        # 提取平均利用率用于概览
        df['AvgUtilization'] = df['Utilization'].apply(np.mean)
        df['StdUtilization'] = df['Utilization'].apply(np.std)  # 负载均衡指标

        csv_path = os.path.join(exp_dir, "summary_data.csv")
        # 临时删除 Utilization 列以保存 CSV（因为它是 list 类型）
        df.drop(columns=['Utilization']).to_csv(csv_path, index=False)
        print(f"Summary CSV saved to {csv_path}")

        # --- 绘图部分的修正 ---

        # 3. 绘图：总成本箱线图
        plt.figure(figsize=(10, 6))
        # 修正：添加 hue="Algorithm", legend=False 以消除 FutureWarning
        sns.boxplot(x="Algorithm", y="TotalCost", data=df, hue="Algorithm", palette="Set2", legend=False)
        plt.title("Total Cost Comparison")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(exp_dir, "cost_comparison.png"), dpi=300)
        plt.close()  # 修正：显式关闭图像，解决 "More than 20 figures have been opened" 警告

        # 4. 绘图：平均利用率对比
        plt.figure(figsize=(10, 6))
        # 修正：palette="Viridis" -> palette="viridis" (全小写)，并添加 hue
        sns.barplot(x="Algorithm", y="AvgUtilization", data=df, hue="Algorithm", errorbar='sd', palette="viridis",
                    legend=False)
        plt.title("Average BSS Utilization")
        plt.ylabel("Utilization Ratio")
        plt.savefig(os.path.join(exp_dir, "utilization_comparison.png"), dpi=300)
        plt.close()  # 修正：显式关闭图像

        # 5. 绘图：BSS 负载均衡性 (利用率标准差，越低越好)
        plt.figure(figsize=(10, 6))
        # 修正：添加 hue
        sns.barplot(x="Algorithm", y="StdUtilization", data=df, hue="Algorithm", palette="Reds", legend=False)
        plt.title("Load Unbalance (Std Dev of Utilization)")
        plt.ylabel("Std Dev")
        plt.savefig(os.path.join(exp_dir, "load_balance_comparison.png"), dpi=300)
        plt.close()  # 修正：显式关闭图像

        # 6. 绘图：Best Response 比例 (纳什均衡稳定性)
        plt.figure(figsize=(10, 6))
        # 修正：添加 hue
        sns.violinplot(x="Algorithm", y="BestResponseCount", data=df, hue="Algorithm", palette="Blues", legend=False)
        plt.title("Nash Equilibrium Stability (Number of Best Response EVs)")
        plt.savefig(os.path.join(exp_dir, "ne_stability.png"), dpi=300)
        plt.close()  # 修正：显式关闭图像


if __name__ == "__main__":
    runner = ExperimentRunner()

    # 定义实验参数（小规模）
    params_small = {
        "EV_num": 20,
        "BSS_num": 5,
        "distribution": "uniform",
        "max_Battery_num": 10
    }

    # 运行
    runner.run_comprehensive_benchmark(params_small, repeat_times=10)

    # 如果想跑大规模，解开下面注释
    # params_large = {
    #     "EV_num": 120,
    #     "BSS_num": 15,
    #     "distribution": "uniform",
    #     "max_Battery_num": 15
    # }
    # runner.run_comprehensive_benchmark(params_large, repeat_times=5)
