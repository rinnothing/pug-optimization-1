import common

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gradient_descent(fun, grad, step, stop, x):
    """
    makes gradient descent using given operators
    :param fun: is a function that we are optimizing
    :param grad: is a function that calculates gradient for fun in a given point
    :param step: is a function that calculates the step based on function, gradient value and current state
    :param stop: is a function that determines when to stop
    :param x: is a point from which we start optimizing
    :return:
    """
    res = common.StateResult()
    # putting function
    res.function = fun

    # putting initial value as first guess
    res.add_guess(x)

    # doing steps until the end
    while not stop(res):
        # calculate gradient
        grad_val = grad(fun, x)

        # calculate step
        h = step(fun, grad_val, state=res)

        # doing a step
        x -= grad_val * h
        res.add_guess(x)

        # storing context information
        res.add_history(np.array([grad_val, h]))

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

    def step(fun, grad_val):
        return h_constant

    return step


def get_partial_stuck_step(eps, h0, divisor=2):
    """
    starts with h0 and when finds itself stuck divides step by divisor
    :param eps: distance between "stuck" values
    :param h0: step to begin from
    :param divisor: divisor to divide step by
    :return: scheduler with given hyperparameters
    """
    h = h0

    def step(fun, grad_val, state: common.StateResult):
        nonlocal h
        # if we return to the same place - change h
        if len(state.guesses) >= 3:
            prev, last = state.guesses[-3], state.guesses[-1]
            dist = np.linalg.norm(prev - last)

            if dist < eps:
                h /= divisor

        return h

    return step


def get_exp_decay_step(h0, l):
    """
    returns exponential decay scheduler
    :param h0: step to begin with
    :param l: exp power
    :return: scheduler with given hyperparameters
    """
    inv_exp = np.exp(-l)
    h = h0 / inv_exp

    def step(fun, grad_val, state):
        nonlocal h

        h *= inv_exp
        return h

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

    def step(fun, grad_val, state):
        nonlocal k

        k += 1
        return h0 * np.power((b * k + 1), a)

    return step


def get_inv_root_step(h0):
    """
    return inv root scheduler
    :param h0: step to begin with
    :return: scheduler with given hyperparameters
    """
    k = -1

    def step(fun, grad_val, state):
        nonlocal k

        k += 1
        return h0 / np.sqrt(k + 1)

    return step
