import math
import random
import optimize.multidim_gradient as mgr
import  common.tests_function as t_fun

import numpy as np

import common


def simulated_annealing(fun, start_point, get_next, temp, cooling_rate, min_temp, max_count):
    x = start_point

    res = common.StateResult()
    res.function = fun
    res.add_guess(x)

    value = fun(x)
    res.add_function_call()
    min_point = x
    min_value = value

    count = 0
    while temp > min_temp and max_count > count:
        count += 1
        next_x = get_next(x)
        next_value = fun(next_x)
        res.add_function_call()

        delta_energy = next_value - value

        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
            x = next_x
            res.add_guess(x)
            value = next_value
            if value < min_value:
                min_point = x
                min_value = value

        temp *= cooling_rate

    res.add_guess(min_point)
    res.success = True
    return res


def get_next_in_sqrt(step_size=0.1):
    def fun(x):
        return np.array([x[0] + random.uniform(-step_size, step_size), x[1] + random.uniform(-step_size, step_size)])
    return fun

if __name__ == "__main__":
    for numb_fun in range(0, 4):
        x = t_fun.functions_with_local_min_2d[numb_fun].lim[1] / 2
        y = x
        len_xy = t_fun.functions_with_local_min_2d[numb_fun].lim[1] - t_fun.functions_with_local_min_2d[numb_fun].lim[0]

        res = simulated_annealing(
            t_fun.functions_with_local_min_2d[numb_fun].function,
            [x, y],
            get_next_in_sqrt(len_xy / 20),
            100,
            0.95,
            1e-2,
            10000
        )
        res1 = t_fun.functions_with_local_min_2d[numb_fun].function(res.get_res())
        res2 = t_fun.functions_with_local_min_2d[numb_fun].function(t_fun.functions_with_local_min_2d[numb_fun].point_min)
        print(res.get_res(), "but need ", t_fun.functions_with_local_min_2d[numb_fun].point_min)
        print(res1, " | ", res2)
        mgr.visualiser_2d(res,  t_fun.functions_with_local_min_2d[numb_fun].lim,  t_fun.functions_with_local_min_2d[numb_fun].lim )
