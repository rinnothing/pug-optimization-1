import random
import time

import numpy as np

import optimize.golden_search
import optimize.random_search
import common.tests_function
import common.tests_function as test_fun
import optimize.gradient_descent as gr
import optimize.multidim_gradient as mgr
from functools import partial
import optimize.simulated_annealing as sim_an
import optimize.genetic as gen


class tests_sim:
    def __init__(self, name, start, vis):
        self.name = name
        self.start = start
        self.vis = vis
        self.need_res: list = []
        self.predicts: list = []
        self.starts: list = []
        self.count: list = []
        self.time: list = []
        self.min: list = []

    def add_in_last_predicate(self, v):
        self.predicts[-1] += v

    def add_in_last_starts(self, v):
        self.starts[-1] += v

    def add_in_last_time(self, v):
        self.time[-1] += v

    def add_in_last_min(self, v):
        self.min[-1] = min(self.min[-1], v)

    def add_new_tests(self, count, need_res):
        self.predicts.append(0)
        self.starts.append(0)
        self.time.append(0)
        self.min.append(1e9)
        self.need_res.append(need_res)
        self.count.append(count)


def start_grad_wolfe(t_fun, start, lim):
    return gr.gradient_descent(t_fun.function, t_fun.gradient,
                               partial(gr.get_next_wolfe, c1=0.388, c2=0.9337, max_count=21),
                               gr.get_stop_x_eps(1.64949e-8), start, max_count=2312)


def start_sim_an(t_fun, start, lim):
    return sim_an.simulated_annealing(t_fun.function, start, sim_an.get_next_in_sqrt((lim[1] - lim[0]) / 40), 7728,
                                      0.98, 0.011, 9311)


def start_gen(t_fun, start, lim):
    return gen.genetic_algorithm(t_fun.function, lim, 54, 80,
                                 10, 0.6342, 0.79, (lim[1] - lim[0]) / 60)


tests_fun = [
    #tests_sim("grad_wolfe", start_grad_wolfe, mgr.visualiser_2d),
    #tests_sim("simulated_annealing", start_sim_an, mgr.visualiser_2d),
    tests_sim("genetic", start_gen, gen.visualiser_2d_gif)
]
print("Test 2d with one min")
count2 = 4
count3 = 0
for t_fun in test_fun.functions_with_local_min_2d:
    lim = t_fun.lim
    count2 -= 1
    count3 += 1
    print(count2)
    for f in tests_fun:
        f.add_new_tests(100, t_fun.function(t_fun.point_min))
    for i in range(100):
        x = random.uniform(lim[0], lim[1])
        y = random.uniform(lim[0], lim[1])
        for f in tests_fun:
            start_time = time.time()
            res = f.start(t_fun, np.array([x, y]), t_fun.lim)
            end_time = time.time()
            f.add_in_last_starts(res.count_of_function_calls)
            f.add_in_last_predicate(t_fun.function(res.get_res()))
            f.add_in_last_time(end_time - start_time)
            f.add_in_last_min(t_fun.function(res.get_res()))
            #gen.visualiser_2d_gif(res, lim_x=lim, lim_y=lim, index=count3)
            # mgr.visualiser_3d(res, lim_x=lim, lim_y=lim, index = count3)

for f in tests_fun:
    print(f.name)
    for i in range(len(f.count)):
        print(f.name, "		func numb:	", i + 1, "		sum_min_value:	", f.predicts[i] / f.count[i],
              "	count iteration:	",
              f.starts[i] / f.count[i], "	Главный минимум:	", f.min[i],
              f"	Время выполнения:	{f.time[i]:.4f}		секунд",
              "	Погрешность среднего:	", abs(f.predicts[i] / f.count[i] - f.need_res[i]),
              "	Погрешность главного:	", abs(f.min[i] - f.need_res[i]))

print("---------------------")
for f in tests_fun:
    for i in range(len(f.count)):
        print(f.name, "	", i + 1, "	", f.predicts[i] / f.count[i],
              "	",
              f.starts[i] / f.count[i], "	", f.min[i],
              f"	{f.time[i]:.4f}",
              "	", abs(f.predicts[i] / f.count[i] - f.need_res[i]),
              "	", abs(f.min[i] - f.need_res[i]))
