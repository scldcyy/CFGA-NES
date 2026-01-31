import copy
import random

import matplotlib.pyplot as plt
import numpy as np

# 导入 CFGA 以使用 init_pop (现在需要传参)
from CFGA import init_pop
from utils.parser_config import param


def NE_Seeking(problem, I=100, init_bts=None):
    """
    NES (Nash Equilibrium Seeking) 算法
    参数:
        problem: BSSProblem 实例
        I: 迭代次数
        init_bts: 初始策略 (如果不提供，则调用 init_pop 随机生成)
    """
    NE_result = {
        "best_solution": [],
        "best_fit": [],
        "best_sum_fit": 0,
        "iter_fit": [],
        "iter_sum_fit": [],
    }

    # 如果没有提供初始解，随机初始化一个
    if init_bts is None:
        # 注意：这里调用 CFGA 的 init_pop 需要传入 problem
        init_bts = init_pop(problem, pop_size=1)[0]

    x_cp = copy.deepcopy(init_bts)
    iter_fit = []

    # 迭代寻找纳什均衡
    for i in range(I):
        order = list(range(problem.N))
        random.shuffle(order)

        # 逐个EV更新策略
        for n in order:
            # 调用 problem 中的方法
            new_strategy = problem.best_response_strategy(n, x_cp)
            x_cp[n] = new_strategy

        # 计算当前迭代的总成本
        costs, _ = problem.f(x_cp)
        iter_fit.append(np.sum(costs))

    # 记录结果
    NE_result["best_solution"] = x_cp
    NE_result["best_fit"] = iter_fit[-1]  # 最后一次迭代的成本

    final_costs, _ = problem.f(x_cp)
    NE_result["best_sum_fit"] = np.sum(final_costs)
    NE_result["iter_fit"] = iter_fit  # 注意：这里为了简化，只存了总和，如果需要存每代详细信息需修改逻辑
    NE_result["iter_sum_fit"] = iter_fit  # 这里和上面一致

    # 绘图部分（保持原有风格，但增加了维度判断防止报错）
    # 如果 iter_fit 是简单的列表，直接 plot 即可
    plt.figure()
    plt.plot(iter_fit)
    plt.title("NES Iteration Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Total Cost")
    # plt.show() # 建议在脚本中控制 show，这里可以注释掉以免阻塞批量运行

    # 打印初始状态和最终状态的最佳响应数量
    # 使用 problem.total_best_response
    init_br_count = problem.total_best_response(init_bts)
    final_br_count = problem.total_best_response(x_cp)
    print(f'init BR count: {init_br_count}/{problem.N}, after NES BR count: {final_br_count}/{problem.N}')

    return NE_result


if __name__ == "__main__":
    # 同样需要手动构造环境进行测试
    from utils.generate_exp_enviroment import EnvironmentManager
    from objective import BSSProblem

    # 构造测试环境
    env_manager = EnvironmentManager()
    env_params = {"EV_num": 20, "BSS_num": 5, "rand_seed": 3407}
    env_data = env_manager.generate_environment(env_params)
    problem = BSSProblem(env_data["data"])

    I = param["NES_param"]["I"]

    # 测试1：从随机生成开始
    print("Testing NES with random initialization...")
    NE_result = NE_Seeking(problem, I=I)

    # 测试2：从文件读取（如果需要模拟完整流程）
    # init_bts = json.load(open("GA_result_test.json", 'r'))["best_solution"]
    # NE_result_2 = NE_Seeking(problem, I=I, init_bts=init_bts)
