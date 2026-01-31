import json
import sys
import random

from tqdm import tqdm

from objective import (SoC_hats,E_init,ln,delta_yita_a,En,SoC_thrs,
                       E_battery_init,bts,N,set_all,objective,f)
from utils.parser_config import get_GA_param
import copy
import numpy as np




def init_pop(pop_size=100):
    pop = []
    for g in range(pop_size * 100):
        if len(pop) == pop_size:
            break
        bts_copy = copy.deepcopy(bts)
        random.shuffle(bts_copy)
        sq = []
        for ev in range(N):
            for bt in bts_copy:
                k, j = bt
                Soc_nk = E_init[ev] - ln[ev, k] * delta_yita_a / En
                if Soc_nk >= SoC_hats[ev] and E_battery_init[k][j] >= SoC_thrs[ev]:
                    sq.append(bt)
                    bts_copy.remove(bt)
                    break
            else:
                # print(ev)
                break
        if len(sq) == N:
            pop.append(sq)
    if len(pop) != pop_size:
        sys.exit(f"种群大小不足,大小为:{len(pop)}")
    return pop


def rank(pop, fitness):
    pop_rank = np.argsort(-fitness)
    rank_pop = [pop[r] for r in pop_rank]
    rank_fit = fitness[pop_rank]
    return rank_pop, rank_fit


def select(pop, fitness, eliteSize, pop_size):
    fit_p = fitness / np.sum(fitness)
    select_pop = pop[:eliteSize] + random.choices(pop, weights=fit_p, k=pop_size - eliteSize)
    return select_pop


def cross(p1, p2, crossRate=0.8):
    if random.random() < crossRate:
        start, end = np.sort(np.random.choice(np.arange(N), 2, False))
        c1 = p1[start:end]
        for gene in p2[end:] + p2[:end]:
            if gene not in c1:
                c1.append(gene)
            if len(c1) == N:
                break
        c1 = c1[-start:] + c1[:-start]

        c2 = p2[start:end]
        # for gene in np.append(select_pop[r2][end:], select_pop[r2][:end]):
        # for gene in select_pop[r2]:
        for gene in p1[end:] + p1[:end]:
            if gene not in c2:
                c2.append(gene)
            if len(c2) == N:
                break
        c2 = c2[-start:] + c2[:-start]
    else:
        c1, c2 = p1.copy(), p2.copy()
    return c1, c2


def mutate(c, mutationRate):
    set_new_x = set(c)
    replace = list(set_all - set_new_x)
    random.shuffle(replace)
    d = 0
    for j in range(len(c)):
        if random.random() < mutationRate and d<len(replace):
            c[j] = replace[d]
            d += 1
        if random.random() < mutationRate:
            a = random.randint(0, N - 1)
            c[a], c[j] = c[j], c[a]
    return c


def CFGA(pop_size=500, maxFEs=500000, eliteSize=50, mutationRate=0.01, crossRate=0.8, obj=objective):
    # pop = [random.sample(bts, N) for _ in range(pop_size)]
    pbar = tqdm(total=maxFEs)
    GA_result = {
        "best_solution": [],
        "best_fit": [],
        "best_sum_fit": 0,
        "iter_fit": [],
        "iter_sum_fit": [],
    }
    pop = init_pop()
    fitness = np.array([obj(x) for x in pop])
    FEs = pop_size
    while FEs < maxFEs:
        rank_pop, rank_fit = rank(pop, fitness)
        pop = select(rank_pop, rank_fit, eliteSize, pop_size)
        for i in range(eliteSize, pop_size, 2):
            p1, p2 = pop[i], pop[i + 1]
            c1, c2 = cross(p1, p2, crossRate)
            c1 = mutate(c1, mutationRate)
            c2 = mutate(c2, mutationRate)
            pop[i], pop[i + 1] = c1, c2
        fitness = np.array([obj(x) for x in pop])
        FEs += pop_size
        best_idx = np.argmax(fitness)
        GA_result["best_solution"] = pop[best_idx]
        best_fit, _ = f(pop[best_idx])
        GA_result["best_fit"] = best_fit.tolist()
        GA_result["best_sum_fit"] = sum(GA_result["best_fit"])
        GA_result["iter_fit"].append(GA_result["best_fit"])
        GA_result["iter_sum_fit"].append(GA_result["best_sum_fit"])
        pbar.set_postfix(bar_format=f"FEs:{FEs},fit:{fitness[best_idx]}, x:{pop[best_idx]}")
        pbar.update(pop_size)
    pbar.close()
    return GA_result


if __name__ == "__main__":
    param = get_GA_param()
    random.seed(param["random_seed"])
    pop_size = param["pop_size"]
    maxFEs = param["maxFEs"]
    eliteSize = param["eliteSize"]
    mutationRate = param["mutationRate"]
    crossRate = param["crossRate"]
    GA_result = CFGA(pop_size=pop_size, maxFEs=maxFEs, eliteSize=eliteSize,
                     mutationRate=mutationRate, crossRate=crossRate, obj=objective)
    json.dump(GA_result, open("exp/exp0005/GA_result.json", "w"), indent=4)


