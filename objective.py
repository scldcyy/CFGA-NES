import copy

import numpy as np


class BSSProblem:
    def __init__(self, env_data):
        """初始化问题模型，传入环境数据"""
        self.N = env_data["EV_num"]
        self.K = env_data["BSS_num"]
        self.E_init = np.array(env_data["EV_SoC"][0])
        self.SoC_hats = np.array(env_data["EV_SoC"][1])
        self.SoC_thrs = np.array(env_data["EV_SoC"][2])
        self.Jk = np.array(env_data["Battery_nums"])

        # 扁平化电池索引
        self.E_battery_init = env_data["Inital_battery_Soc"]
        self.ln = np.array(env_data["l_n_k"])

        # 常量定义
        sa = 50
        self.delta_yita_a = 7.344 * 1e-5 * sa ** 2 - 7.656 * 1e-3 * sa + 0.3536
        self.En = 75
        self.Lk_max = self.Jk * self.En
        self.Rk = 12.5 * self.Jk
        self.p_grid = 0.85

        # 预计算每个BSS的初始电量总和
        self.Lkavi_pre = np.array([np.sum(np.array(self.E_battery_init[k]) * self.En) for k in range(self.K)])

        # 生成所有候选解集合 (k, j)
        self.bts = []
        for i in range(len(self.Jk)):
            for j in range(self.Jk[i]):
                self.bts.append((i, j))
        self.set_all = set(self.bts)

        # 权重系数
        self.alpha = 0.5
        self.beta = 0.5
        self.tao = 0.6

    def f(self, x):
        """
        向量化计算目标函数
        x: list of tuples [(k1, j1), (k2, j2), ...]
        Returns: (costs, punnishs)
        """
        x_arr = np.array(x)  # Shape (N, 2)
        ks = x_arr[:, 0]
        # js = x_arr[:, 1] # 暂时不用 j 的索引计算

        # 1. 计算行驶能耗后剩余电量
        dists = self.ln[np.arange(self.N), ks]
        E_res = self.E_init - dists * self.delta_yita_a / self.En

        # 2. 获取目标电池初始电量
        E_bat_init_vals = np.array([self.E_battery_init[k][j] for k, j in x])

        # 3. 计算交换电量 en
        en = (E_bat_init_vals - E_res) * self.En

        # 4. 计算每个 BSS 的总负载 Lk
        Lk = np.zeros(self.K)
        np.add.at(Lk, ks, en)

        # 5. 计算约束惩罚
        violation = (E_res < self.SoC_hats) | (E_bat_init_vals <= self.SoC_thrs)
        punnishs = np.where(violation, 1e4, 0.0)

        # 6. 计算电价 pk
        Lkavi_cur = self.Lkavi_pre + self.Rk
        Gammak = 2 * self.p_grid - Lkavi_cur / self.Lk_max
        Lambdak = 1 / self.Lk_max
        pk_all = Gammak + Lambdak * Lk

        # 获取每个 EV 面对的电价
        pk_selected = pk_all[ks]

        # 7. 计算最终成本
        costs = self.alpha * en * pk_selected + self.beta * self.tao * dists

        return costs, punnishs

    def objective(self, x):
        costs, punnishs = self.f(x)
        total_punnish = np.sum(punnishs)
        if total_punnish > 0:
            return 1e-10
        return 1.0 / np.sum(costs)

    def potential_function(self, x):
        """
        计算势函数值 (Potential Function)，用于验证博弈性质。
        完全还原原始 objective.py 中的逻辑。
        """
        x_arr = np.array(x)
        ks = x_arr[:, 0]

        # 复用逻辑计算中间变量
        dists = self.ln[np.arange(self.N), ks]
        E_res = self.E_init - dists * self.delta_yita_a / self.En
        E_bat_init_vals = np.array([self.E_battery_init[k][j] for k, j in x])
        en = (E_bat_init_vals - E_res) * self.En

        Lk = np.zeros(self.K)
        Qk = np.zeros(self.K)  # 平方和项，原始代码逻辑
        np.add.at(Lk, ks, en)
        np.add.at(Qk, ks, en * en)

        violation = (E_res < self.SoC_hats) | (E_bat_init_vals <= self.SoC_thrs)
        punnishs = np.where(violation, 1e4, 0.0)

        Lkavi_cur = self.Lkavi_pre + self.Rk
        Gammak = 2 * self.p_grid - Lkavi_cur / self.Lk_max
        Lambdak = 1 / self.Lk_max
        pk_all = Gammak + Lambdak * Lk
        pk_selected = pk_all[ks]

        costs = self.alpha * en * pk_selected + self.beta * self.tao * dists

        # 计算 Potential
        # 原始公式: potential+=alpha* (Gammak[k]*Lk[k]+0.5*Lambdak[k]*(Lk[k]*Lk[k]+Qk[k]))+beta*tao*ln[n,k]
        # 这里进行向量化计算
        term1 = self.alpha * (Gammak[ks] * Lk[ks] + 0.5 * Lambdak[ks] * (Lk[ks] ** 2 + Qk[ks]))
        term2 = self.beta * self.tao * dists
        potential = np.sum(term1 + term2)

        return costs, punnishs, potential

    def total_best_response(self, x):
        """
        计算当前策略 x 中有多少个 EV 处于最佳响应状态。
        Ret: 处于最佳响应的 EV 数量
        """
        ret = len(x)
        cost, punnish = self.f(x)

        # 为了避免修改传入的 x，先深拷贝一份用于尝试
        x_cp = copy.deepcopy(x)

        # 遍历每个 EV
        for i in range(len(x)):
            current_cost = cost[i]
            current_punnish = punnish[i]

            # 当前策略如果是无效的（有惩罚），那肯定不是最佳响应（除非所有策略都无效）
            # 原始代码逻辑：如果找到更好的，或者当前本来就有惩罚，就尝试找更好的

            cand_set = list(self.set_all - {tuple(x[i])})

            # 这里的逻辑是：只要发现一个更好的策略，就说明当前不是最佳响应，立即 break 检查下一个 EV
            found_better = False

            # 原始逻辑是逐个尝试所有其他策略。这在 set_all 很大时非常慢。
            # 但为了保持一致性，我们照搬逻辑。
            # 优化点：其实不需要每次都重新计算所有人的 cost，只算变动的那个人的即可，
            # 但由于电价是耦合的，一个人的变动会影响其他人，
            # 不过我们只关心第 i 个人的 cost 是否降低。

            # 为了加速，我们可以只针对第 i 个人进行比较。
            # 但由于 self.f 是批量计算的，这里直接调用 f 比较方便，尽管开销大。

            original_choice = x_cp[i]
            for s in cand_set:
                x_cp[i] = s  # 尝试新策略
                new_costs, new_punnishs = self.f(x_cp)

                # 判定条件：成本降低且无惩罚
                if (new_costs[i] < current_cost and new_punnishs[i] == 0) or (
                        current_punnish > 0 and new_punnishs[i] == 0):
                    ret -= 1
                    found_better = True
                    break

                # 如果当前本来就有惩罚，只要找到个惩罚更小的（这里简化为0）就算更好
                # 原始代码逻辑：if (new_cost[i] < cost[i] and new_punnish[i] == 0) or punnish[i] > 0:
                # 注意：原始代码中 `punnish[i] > 0` 这个条件有点宽泛，意味着只要当前有惩罚，
                # 不管新策略好不好，ret都减1？
                # 细看原始代码:
                # if (new_cost[i] < cost[i] and new_punnish[i] == 0) or punnish[i] > 0:
                #    ret -= 1; break
                # 这意味着：如果当前有惩罚，直接认为不是最佳响应（假设总存在无惩罚解）。
                # 或者如果有更优解，也不是最佳响应。

            x_cp[i] = original_choice  # 恢复策略，准备检查下一个人

        return ret

    def best_response_strategy(self, ev_num, x):
        """
        为第 ev_num 号 EV 寻找最佳响应策略。
        """
        x_cp = copy.deepcopy(x)
        cand_set = list(self.set_all - {tuple(x_cp[ev_num])})

        # 当前状态
        best_s = x_cp[ev_num]
        costs, punnishs = self.f(x)
        min_cost = costs[ev_num]
        is_punnished = punnishs[ev_num] > 0

        for s in cand_set:
            x_cp[ev_num] = s
            new_costs, new_punnishs = self.f(x_cp)

            # 更新最佳策略逻辑
            # 1. 如果新策略无惩罚，且成本更低 -> 更新
            # 2. 如果当前策略有惩罚，且新策略无惩罚 -> 更新
            if new_punnishs[ev_num] == 0:
                if is_punnished or new_costs[ev_num] < min_cost:
                    min_cost = new_costs[ev_num]
                    best_s = s
                    is_punnished = False  # 找到了无惩罚的，状态更新

        return best_s
