import random
import time

import numpy as np

import optimize.golden_search
import optimize.random_search
import common.tests_function
import optimize.gradient_descent as gr
import common.tests_function as test_fun
import optimize.multidim_gradient as mgr
from functools import partial
import optimize.bfgs as bfgs
# import optimize.newton as newton


class test_more_fun:
    def __init__(self, name, function, grad):
        self.name = name
        self.function = function
        self.grad = grad
        self.need_res: list = []
        self.predicts: list = []
        self.starts: list = []
        self.starts_grad: list = []
        self.hess: list = []
        self.count: list = []
        self.time: list = []
        self.min: list = []

    def add_in_last_predicate(self, v):
        self.predicts[-1] += v

    def add_in_last_starts(self, v):
        self.starts[-1] += v

    def add_in_last_starts_grad(self, v):
        self.starts_grad[-1] += v

    def add_in_last_starts_hess(self, v):
        self.hess[-1] += v

    def add_in_last_time(self, v):
        self.time[-1] += v

    def add_in_last_min(self, v):
        self.min[-1] = min(self.min[-1], v)

    def add_new_tests(self, count, need_res):
        self.predicts.append(0)
        self.starts.append(0)
        self.starts_grad.append(0)
        self.hess.append(0)
        self.time.append(0)
        self.min.append(1e9)
        self.need_res.append(need_res)
        self.count.append(count)


def run_func(fun, grad, get_next, stop = gr.get_stop_f_eps(1e-6), point = np.array([0, 0])):
    return gr.gradient_descent(fun, grad, get_next, stop, point, max_count=10000)

def run_bfgs_func(fun, grad, get_next, stop=gr.get_stop_x_eps(1e-6), point=np.array([0, 0])):
    return optimize.bfgs.bfgs(fun, grad,
                  partial(gr.get_next_wolfe, c1=0.013, c2=0.887, max_count = 10),
                  gr.get_stop_x_eps(1.5e-8), np.array([x, y]), max_count = 8000)
def run_bfgs_func_2(fun, grad, get_next, stop=gr.get_stop_x_eps(1e-6), point=np.array([0, 0])):
    return optimize.bfgs.bfgs(fun, grad,
                  gr.get_next_wolfe,
                  gr.get_stop_f_eps(1e-6), np.array([x, y]), max_count = 10000)
# def run_newton_func(fun, grad, hess, get_next, stop=gr.get_stop_x_eps(1e-6), point=np.array([0, 0])):
#     return optimize.newton.newton(fun, grad, hess, get_next, stop, point, max_count=1000)


def default_grad(fun):
    return fun.gradient


def sym_grad(fun):
    return gr.create_grad_from_bad_func(fun.function, gr.symmetric_derivative)


def forward_grad(fun):
    return gr.create_grad_from_bad_func(fun.function, gr.forward_derivative)


def back_grad(fun):
    return gr.create_grad_from_bad_func(fun.function, gr.backward_derivative)


tests_fun_bfgs = [
    test_more_fun("bfgs_wolf", gr.get_next_wolfe, default_grad)
]
tests_fun_newton = [
    test_more_fun("newton_wolf", gr.get_next_wolfe, default_grad)
]

tests_fun = [
    test_more_fun("wol_def", gr.get_next_wolfe, default_grad),
]
'''TEST TO STANDART'''
# gh= 0
# for i in range(0, 100):
#     gh+=random.uniform(-10, 10)
# print(gh)
# print(" ")
# print("Test 2d with local min")
# count2 = 4
# for t_fun in test_fun.functions_with_one_min_2d:
#     lim = t_fun.lim
#     count2 -= 1
#     print(count2)
#     for f in tests_fun:
#         f.add_new_tests(1, t_fun.function(t_fun.point_min))
#     for _ in range(1):
#         x = random.uniform(lim[0], lim[1])
#         y = random.uniform(lim[0], lim[1])
#         for f in tests_fun:
#             start_time = time.time()
#             res = run_func(t_fun.function, f.grad(t_fun), f.function, point=np.array([x, y]))
#             end_time = time.time()
#             f.add_in_last_starts(res.count_of_function_calls)
#             f.add_in_last_starts_grad(res.count_of_gradient_calls)
#             f.add_in_last_predicate(t_fun.function(res.get_res()))
#             f.add_in_last_time(end_time - start_time)
#             f.add_in_last_min(t_fun.function(res.get_res()))
#             #mgr.visualiser_2d(res, lim_x=lim, lim_y=lim)
#             #mgr.visualiser_3d(res, lim_x=lim, lim_y=lim)

# print(count2)
# for f in tests_fun:
#     print()
#     for i in range(len(f.count)):
#         print(f.name, "		func numb:	", i + 1, "		sum_min_value:	", f.predicts[i] / f.count[i], "	count iteration:	",
#               f.starts[i] / f.count[i], "	count iteration grad:	",
#               f.starts_grad[i] / f.count[i], "	Главный минимум:	", f.min[i], f"	Время выполнения:	{f.time[i]:.4f}		секунд",
#               "	Погрешность среднего:	", abs(f.predicts[i] / f.count[i] - f.need_res[i]),
#               "	Погрешность главного:	", abs(f.min[i] - f.need_res[i]))

'''TEST TO BFGS'''
print(" ")
print("Test 2d with local min")
count2 = 4
count3 = 0
for t_fun in test_fun.functions_with_one_min_2d:
    lim = t_fun.lim
    count2 -= 1
    count3 += 1
    print(count2)
    for f in tests_fun_bfgs:
        f.add_new_tests(1, t_fun.function(t_fun.point_min))
    for _ in range(1):
        x = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
        y = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
        for f in tests_fun_bfgs:
            start_time = time.time()
            res = run_bfgs_func(t_fun.function, f.grad(t_fun), f.function, point=np.array([x, y]))
            res2 = run_bfgs_func_2(t_fun.function, f.grad(t_fun), f.function, point=np.array([x, y]))
            end_time = time.time()
            f.add_in_last_starts(res.count_of_function_calls)
            f.add_in_last_starts_grad(res.count_of_gradient_calls)
            f.add_in_last_starts_hess(res.count_of_hessian_calls)
            f.add_in_last_predicate(t_fun.function(res.get_res()))
            f.add_in_last_time(end_time - start_time)
            f.add_in_last_min(t_fun.function(res.get_res()))
            mgr.visualiser_2d_2_on1(res, res2, lim_x=lim, lim_y=lim, index = count3)
            mgr.visualiser_3d_2_on_1(res, res2, lim_x=lim, lim_y=lim, index = count3)

print(count2)
for f in tests_fun:
    print()
    for i in range(len(f.count)):
        print(f.name, "		func numb:	", i + 1, "		sum_min_value:	", f.predicts[i] / f.count[i],
              "	count iteration:	",
              f.starts[i] / f.count[i], "	count iteration grad:	",
              f.starts_grad[i] / f.count[i], "	count iteration hess:	",
              f.hess[i] / f.count[i], "	Главный минимум:	", f.min[i],
              f"	Время выполнения:	{f.time[i]:.4f}		секунд",
              "	Погрешность среднего:	", abs(f.predicts[i] / f.count[i] - f.need_res[i]),
              "	Погрешность главного:	", abs(f.min[i] - f.need_res[i]))
print("---------------------")
for f in tests_fun:
    for i in range(len(f.count)):
        print(f.name, "	", i + 1, "	", f.predicts[i] / f.count[i],
              "	",
              f.starts[i] / f.count[i], "	",
              f.starts_grad[i] / f.count[i], "	",
              f.hess[i] / f.count[i], "	", f.min[i],
              f"	{f.time[i]:.4f}",
              "	", abs(f.predicts[i] / f.count[i] - f.need_res[i]),
              "	", abs(f.min[i] - f.need_res[i]))

'''TEST TO NEWTON'''
# print(" ")
# print("Test 2d with local min")
# count2 = 4
# for t_fun in test_fun.functions_with_one_min_2d:
#     lim = t_fun.lim
#     count2 -= 1
#     print(count2)
#     for f in tests_fun:
#         f.add_new_tests(1, t_fun.function(t_fun.point_min))
#     for _ in range(1):
#         x = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
#         y = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
#         for f in tests_fun:
#             start_time = time.time()
#             res = run_newton_func(t_fun.function, f.grad(t_fun), f.function, point=np.array([x, y]))
#             end_time = time.time()
#             f.add_in_last_starts(res.count_of_function_calls)
#             f.add_in_last_starts_grad(res.count_of_gradient_calls)
#             f.add_in_last_starts_hess(res.count_of_hessian_calls)
#             f.add_in_last_predicate(t_fun.function(res.get_res()))
#             f.add_in_last_time(end_time - start_time)
#             f.add_in_last_min(t_fun.function(res.get_res()))
#             # mgr.visualiser_2d(res, lim_x=lim, lim_y=lim)
#             # mgr.visualiser_3d(res, lim_x=lim, lim_y=lim)
#
# print(count2)
# for f in tests_fun:
#     print()
#     for i in range(len(f.count)):
#         print(f.name, "		func numb:	", i + 1, "		sum_min_value:	", f.predicts[i] / f.count[i],
#               "	count iteration:	",
#               f.starts[i] / f.count[i], "	count iteration grad:	",
#               f.starts_grad[i] / f.count[i], "	count iteration hess:	",
#               f.hess[i] / f.count[i], "	Главный минимум:	", f.min[i],
#               f"	Время выполнения:	{f.time[i]:.4f}		секунд",
#               "	Погрешность среднего:	", abs(f.predicts[i] / f.count[i] - f.need_res[i]),
#               "	Погрешность главного:	", abs(f.min[i] - f.need_res[i]))
"""

for f in tests_fun:
    sum_p=0
    sum_s=0
    sum_t=0
    size = len(f.count)
    for i in range(size):
        sum_p+=f.predicts[i] / f.count[i]
        sum_s+=f.starts[i] / f.count[i]
        sum_t+=f.time[i]
    print(f.name, "		middle_sum_min_value:	", sum_p/size, "	count iteration:	",
              sum_s/size, f"	Общее время выполнения:		{sum_t:.4f}		секунд		Cреднее время		{sum_t/size:.4f}		")

"""
