import json
import random

from CFGA import init_pop
from objective import *
from utils.parser_config import param
from matplotlib import pyplot as plt


def NE_Seeking(I=100, init_bts=None):
    # NES算法
    NE_result = {
        "best_solution": [],
        "best_fit": [],
        "best_sum_fit": 0,
        "iter_fit": [],
        "iter_sum_fit": [],
    }
    x_cp=copy.deepcopy(init_bts)
    # print("初始策略：", sum(f([[q[1], q[2]] for q in Q])))
    iter_fit=[]
    for i in range(I):
        order=list(range(N))
        random.shuffle(order)
        for n in order:
            new_strategy=best_response_strategy(n,x_cp)
            x_cp[n]=new_strategy
        iter_fit.append(np.sum(f(x_cp)))
    NE_result["best_solution"] = x_cp
    NE_result["best_fit"] = iter_fit[-1]
    NE_result["best_sum_fit"] = np.sum(f(x_cp))
    NE_result["iter_fit"] = iter_fit
    NE_result["iter_sum_fit"] = np.sum(iter_fit, axis=1).tolist()
    tmp=np.array(iter_fit).T
    for line in tmp:
        plt.plot(line)
    plt.show()
    print(f'init: {total_best_response(init_bts)}, after NES: {total_best_response(x_cp)}')
    return NE_result


if __name__ == "__main__":
    I = param["NES_param"]["I"]
    init_bts=json.load(open("exp/exp0005/GA_result.json", 'r'))
    init_bts=list(map(tuple,init_bts["best_solution"]))
    init_bts=init_pop(1)[0]
    NE_result = NE_Seeking(I=I,init_bts=init_bts)
    pass
