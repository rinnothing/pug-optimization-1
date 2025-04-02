import random
import common.tests_function
import optimize.general_visualiser as vis

import common

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def random_search(fun, max_count_step=1000, stop=None, bounds=None):
    res = common.StateResult()
    # putting fun
    res.function = fun

    # need to have bounds for random search
    if bounds is None:
        return res

    # init bounds
    s, f = bounds

    # create lasp point
    last_point = random.uniform(s, f)
    fun_last = fun(last_point)
    res.add_function_call()

    # you can put values you need to explain previous steps in history
    res.add_history(np.array([s, f]))
    res.add_guess(s)
    res.add_history(np.array([s, f]))
    # doing random search
    while not stop(res) and max_count_step > 0:
        max_count_step -= 1
        new_point = random.uniform(s, f)
        fun_new = fun(new_point)
        res.add_function_call()
        if fun_new == fun_last:
            res.add_guess(res.guesses[-1])
            res.add_history(res.history[-1])
            continue
        if fun_new < fun_last:
            if new_point < last_point:
                f = last_point
            else:
                s = last_point
            last_point = new_point
            fun_last = fun_new
            my_choose = new_point
        else:
            if new_point < last_point:
                s = new_point
            else:
                f = new_point
            my_choose = last_point
        res.add_guess(my_choose)
        res.add_history(np.array([s, f]))

    res.add_guess((s + f) / 2)

    res.success = True
    return res


def get_eps_stop_determiner(eps: float):
    """
    eps bounds difference stopper, only works for binary search
    :param eps: value, that determines what bounds difference it should stop
    :return: to stop or not
    """

    def determiner(state: common.OptimizeResult) -> bool:
        return (state.history[-1][1] - state.history[-1][0]) < eps

    return determiner

# example on usage of created functions
if __name__ == "__main__":
    for test_func in common.tests_function.functions_with_one_min:
        lim = test_func.lim
        result = random_search(test_func.function, stop=get_eps_stop_determiner(0.1), bounds=lim)
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        vis.visualiser_path(result, lim, 200)

    print("start functions with local min")
    for test_func in common.tests_function.functions_with_local_min:
        lim = test_func.lim
        result = random_search(test_func.function, stop=get_eps_stop_determiner(0.1), bounds=lim)
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        vis.visualiser_path(result, lim, 200)
