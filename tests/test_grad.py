import random
import time

import numpy as np

import optimize.golden_search
import optimize.random_search
import common.tests_function
import optimize.gradient_descent as gr
import common.tests_function as test_fun
import optimize.multidim_gradient as mgr

class test_more_fun:
    def __init__(self, name, function, grad):
        self.name = name
        self.function = function
        self.grad = grad
        self.predicts: list = []
        self.starts: list = []
        self.count: list = []
        self.time: list = []
    def add_in_last_predicate(self, v):
        self.predicts[-1] += v
    def add_in_last_starts(self, v):
        self.starts[-1] += v

    def add_in_last_time(self, v):
        self.time[-1] += v
    def add_new_tests(self, count):
        self.predicts.append(0)
        self.starts.append(0)
        self.time.append(0)
        self.count.append(count)

def run_func(fun, grad, get_next, stop = gr.get_stop_x_eps(0.01), point = np.array([0, 0])):
    return gr.gradient_descent(fun, grad, get_next, stop, point, max_count=100)

def default_grad(fun):
    return fun.gradient

tests_fun = [
    test_more_fun("gol_d", gr.get_next_gold, default_grad),
    test_more_fun("ran_d", gr.get_next_random, default_grad),
    test_more_fun("wol_d", gr.get_next_wolfe, default_grad),
 #   test_more_fun("con_d_1", gr.get_constant_step(1), default_grad),
 #   test_more_fun("con_d_05", gr.get_constant_step(0.5), default_grad),
 #   test_more_fun("con_d_01", gr.get_constant_step(0.1), default_grad),
    test_more_fun("par_d_1", gr.get_partial_stuck_step(0.01, 1), default_grad),
    test_more_fun("par_d_05", gr.get_partial_stuck_step(0.01, 0.5), default_grad),
    test_more_fun("par_d_01", gr.get_partial_stuck_step(0.01, 0.1), default_grad)
]

print("Test 2d with one min")
count2 =4
for t_fun in test_fun.functions_with_one_min_2d:
    lim = t_fun.lim
    count2-=1
    print(count2)
    for f in tests_fun:
        f.add_new_tests(20)
    for _ in range(20):
        x = random.uniform(lim[0], lim[1])
        y = random.uniform(lim[0], lim[1])
        for f in tests_fun:
            start_time = time.time()
            res = run_func(t_fun.function, f.grad(t_fun), f.function, point=np.array([x, y]))
            end_time = time.time()
            f.add_in_last_starts(len(res.guesses))
            f.add_in_last_predicate(res.get_res())
            f.add_in_last_time(end_time - start_time)
           # mgr.visualiser_2d(res, lim_x=lim, lim_y=lim)
            #mgr.visualiser_3d(res, lim_x=lim, lim_y=lim)

print(count2)
for f in tests_fun:
    print(f.name, ": ")
    for i in range(len(f.count)):
        print("func numb: ", i + 1, " sum_min_value: ", f.predicts[i] / f.count[i], " | count iteration: ", f.starts[i] / f.count[i], f" | Время выполнения: {f.time[i]:.4f} секунд")