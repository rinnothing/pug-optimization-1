import optimize.general_visualiser as vis
import common
import common.tests_function
import optimize.wolfe_conditions
import optimize.golden_search
import optimize.random_search

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import optimize.wolfe_conditions


def gradient_descent(fun, grad, get_next, stop, x,  min_x = np.array([-100, -100]), max_x = np.array([100, 100]), min_count = 10, max_count = 100):
    """
    makes gradient descent using given operators
    :param fun: is a function that we are optimizing
    :param grad: is a function that calculates gradient for fun in a given point
    :param get_next: is a function that get next value
    :param stop: is a function that determines when to stop
    :param x: is a point from which we start optimizing
    :return:
    """
    res = common.StateResult()
    # putting function
    res.function = fun

    # putting initial value as first guess
    res.add_guess(x)
    save_min_value = [x, fun(x)]
    # doing steps until the end
    count = 0
    while (not stop(res) or min_count > count) and max_count > count:
        count+=1
        # calculate gradient
        antigrad_val = -1 * grad(x)
        if np.linalg.norm(antigrad_val) == 0:
            antigrad_val = 1/count
        x = get_next(res, fun, grad, antigrad_val, x)
        fun_x = fun(x)
        if fun_x < save_min_value[1]:
            save_min_value = [x, fun_x]
        for i in range(len(x)):
            x[i] = min(x[i], max_x[i])
            x[i] = max(x[i], min_x[i])

        res.add_guess(x)

    res.count_of_function_calls = count

    res.add_guess(save_min_value[0])
    res.success = True
    return res


# if you know derivative explicitly you can use it, but in other cases take a look at those functions
def forward_derivative(fun, x, delta=1.0):
    """
    calculates forward derivative
    :param fun: function to calculate derivative
    :param x: point in which to calculate
    :param delta: which delta to use (1.0 by default)
    :return: derivative
    """
    x_1 = x + np.ones_like(x) * delta

    return (fun(x_1) - fun(x)) / delta


def backward_derivative(fun, x, delta=1.0):
    """
    calculates backward derivative
    :param fun: function to calculate derivative
    :param x: point in which to calculate
    :param delta: which delta to use (1.0 by default)
    :return: derivative
    """
    x_1 = x - np.ones_like(x) * delta

    return (fun(x) - fun(x_1)) / delta


def symmetric_derivative(fun, x, delta=1.0):
    """
    calculates symmetric derivative
    :param fun: function to calculate derivative
    :param x: point in which to calculate
    :param delta: which delta to use (1.0 by default)
    :return: derivative
    """
    x_1 = x + np.ones_like(x) * delta
    x_2 = x - np.ones_like(x) * delta

    return (fun(x_1) - fun(x_2)) / (2 * delta)


# calculating steps using scheduling
def get_constant_step(h_constant):
    """
    the simplest scheduler, always returns constant step
    :param h_constant: the step to return
    :return: scheduler with given hyperparameters
    """

    def step(state, fun, grad, antigrad_val, x):
        return x + h_constant * antigrad_val

    return step


def get_partial_stuck_step(eps, h0, divisor=2.0):
    """
    starts with h0 and when finds itself stuck divides step by divisor
    :param eps: distance between "stuck" values
    :param h0: step to begin from
    :param divisor: divisor to divide step by
    :return: scheduler with given hyperparameters
    """
    h = h0

    def step(state, fun, grad, antigrad_val, x):
        nonlocal h
        # if we return to the same place - change h
        if len(state.guesses) >= 3:
            prev, last = state.guesses[-3], state.guesses[-1]
            dist = np.linalg.norm(prev - last)

            if dist < eps:
                h /= divisor

        return x + antigrad_val * h

    return step


def get_exp_decay_step(h0, l):
    """
    returns exponential decay scheduler
    :param h0: step to begin with
    :param l: exp power
    :return: scheduler with given hyperparameters
    """
    inv_exp = np.exp(l)
    h = h0 / inv_exp

    def step(state, fun, grad, antigrad_val, x):
        nonlocal h

        h *= inv_exp
        return x + antigrad_val * h

    return step


def get_pol_decay_step(h0, a, b):
    """
    returns polynomial decay scheduler
    :param h0: step to begin with
    :param a: power
    :param b: step factor
    :return: scheduler with given hyperparameters
    """
    k = -1

    def step(state, fun, grad, antigrad_val, x):
        nonlocal k

        k += 1
        return x + antigrad_val * h0 * np.power((b * k + 1), a)

    return step


def get_inv_root_step(h0):
    """
    return inv root scheduler
    :param h0: step to begin with
    :return: scheduler with given hyperparameters
    """
    k = -1

    def step(state, fun, grad, antigrad_val, x):
        nonlocal k

        k += 1
        return x + antigrad_val * h0 / np.sqrt(k + 1)

    return step


# absolute stoppers
def get_stop_x_eps(eps):
    def stop(state: common.StateResult):
        if len(state.guesses) < 2:
            return False

        return np.linalg.norm(state.guesses[-1] - state.guesses[-2]) < eps

    return stop


def get_stop_f_eps(eps):
    def stop(state: common.StateResult):
        if len(state.guesses) < 2:
            return False

        last_f = state.function(state.guesses[-1])
        prev_f = state.function(state.guesses[-2])

        return np.linalg.norm(last_f - prev_f) < eps

    return stop


def get_eps_stop_determiner(eps: float):
    """
    eps bounds difference stopper, only works for binary search
    :param eps: value, that determines what bounds difference it should stop
    :return: to stop or not
    """

    def determiner(state: common.OptimizeResult) -> bool:
        return (state.history[-1][1] - state.history[-1][0]) < eps

    return determiner


def get_next_gold(state, fun, grad, antigrad_val, x):
    def func_for_gold(y):
        return fun(x + antigrad_val * y)
    res = optimize.golden_search.golden_search(func_for_gold, stop=get_eps_stop_determiner(0.01), bounds=[0, 1])
    return x + antigrad_val * res.get_res()

def get_next_random(state, fun, grad, antigrad_val, x):
    def func_for_random(y):
        return fun(x + antigrad_val * y)
    res = optimize.random_search.random_search(func_for_random, stop=get_eps_stop_determiner(0.01), bounds=[0, 1])
    return x + antigrad_val * res.get_res()

def get_next_wolfe(state, fun, grad, antigrad_val, x):
    res = optimize.wolfe_conditions.wolfe_conditions(fun, grad, antigrad_val/10, x)
    return res.get_res()

def create_grad_from_bad_func(fun, bad_func):
    def grad(x):
        return bad_func(fun, x)
    return grad


# maybe will add other later

if __name__ == "__main__":
    # example
    # f = lambda x: (x ** 4) - 5 * (x ** 2) + 2 * x + 5
    # gr = lambda fun, x: 4 * x ** 3 - 10 * x + 2
    f = lambda x: x ** 4 / 100.0 - x ** 3 / 10 - x ** 2 / 2 + 2 * x + 5
    gr = lambda fun, x: x ** 3 / 25.0 - x ** 2 * 3.0 / 10.0 - x + 2.0

    for test_func in common.tests_function.functions_with_one_min:
        lim = test_func.lim
        result = gradient_descent(test_func.function, create_grad_from_bad_func(test_func.function, symmetric_derivative), get_next_wolfe, get_stop_x_eps(0.01), np.array([lim[0]]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        new_lim = np.array([lim[0] - lim[1] + lim[0], lim[1]])
        new_lim[0] = min(new_lim[0], result.get_res() - 1)
        new_lim[1] = max(new_lim[1], result.get_res() + 1)
        vis.visualiser_path(result, new_lim, 500)
    print("start functions with local min")
    for test_func in common.tests_function.functions_with_local_min:
        lim = test_func.lim
        result = gradient_descent(test_func.function, test_func.gradient, get_next_wolfe, get_stop_x_eps(0.01), np.array([lim[0]]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        new_lim = np.array([lim[0] - lim[1] + lim[0], lim[1]])
        new_lim[0] = min(new_lim[0], result.get_res() - 1)
        new_lim[1] = max(new_lim[1], result.get_res() + 1)
        vis.visualiser_path(result, new_lim, 500)


