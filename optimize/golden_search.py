import common
import common.tests_function
import optimize.general_visualiser as vis

import numpy as np


def golden_search(fun, max_count_step=1000000, stop=None, bounds=None):
    res = common.StateResult()
    # putting fun
    res.function = fun

    # need to have bounds for binary search
    if bounds is None:
        return res

    # init bounds
    s, f = bounds
    # init Golden ratio
    gold = (np.sqrt(5) + 1) / 2

    # create left and right points
    l = f - (f - s) / gold
    r = s + (f - s) / gold
    fun_l = fun(l)
    res.add_function_call()
    fun_r = fun(r)
    res.add_function_call()

    # you can put values you need to explain previous steps in history
    res.add_history(np.array([s, f]))

    # doing golden search
    while not stop(res) and max_count_step:
        max_count_step -= 1
        if fun_l < fun_r:
            f = r
            my_choose = r
            r = l
            fun_r = fun_l
            l = f - (f - s) / gold
            fun_l = fun(l)
            res.add_function_call()
        else:
            s = l
            my_choose = l
            l = r
            fun_l = fun_r
            r = s + (f - s) / gold
            fun_r = fun(r)
            res.add_function_call()

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
        result = golden_search(test_func.function, stop=get_eps_stop_determiner(0.1), bounds=lim)
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        vis.visualiser(result, lim, 200)
    print("start functions with local min")
    for test_func in common.tests_function.functions_with_local_min:
        lim = test_func.lim
        result = golden_search(test_func.function, stop=get_eps_stop_determiner(0.1), bounds=lim)
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        vis.visualiser(result, lim, 200)
