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


class tests_sim:
    def __init__(self, name):
        self.name = name
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


tests_fun = [tests_sim("simulated_annealing")]

def start_grad_wolfe(t_fun):
    return gr.gradient_descent(t_fun.function, t_fun.gradient,
                  partial(gr.get_next_wolfe, c1=0.388, c2=0.9337, max_count = 21),
                  gr.get_stop_x_eps(1.64949e-8), np.array([x, y]), max_count = 2312)

print("Test 2d with one min")
count2 = 4
count3 = 0
for t_fun in test_fun.functions_with_one_min_2d:
    lim = t_fun.lim
    count2 -= 1
    count3 += 1
    print(count2)
    for f in tests_fun:
        f.add_new_tests(1, t_fun.function(t_fun.point_min))
    for i in range(1):
        x = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
        y = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
        for f in tests_fun:
            start_time = time.time()
            res = sim_an.simulated_annealing(t_fun.function,[x, y], sim_an.get_next_in_sqrt((lim[1] - lim[0]) / 40),7728,0.98,0.011,9311)
            #res = start_grad_wolfe(t_fun)
            end_time = time.time()

            f.add_in_last_starts(res.count_of_function_calls)
            f.add_in_last_predicate(t_fun.function(res.get_res()))
            f.add_in_last_time(end_time - start_time)
            f.add_in_last_min(t_fun.function(res.get_res()))
            mgr.visualiser_2d_in(res, lim_x=lim, lim_y=lim, index = count3)
            mgr.visualiser_3d_in(res, lim_x=lim, lim_y=lim, index = count3)

for f in tests_fun:
    print()
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