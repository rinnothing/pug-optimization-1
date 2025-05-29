import math

from autograd import grad
from autograd import hessian

import autograd.numpy as np

class test_function:
    def __init__(self, function, lim, point_min = None):
        self.function = function
        self.gradient = grad(function)

        # there's an error (or a strange feature) in pytorch that calculates one argument functions as (1, 1, 1) arrays
        # this function addresses this issue by an explicit check
        def fixed_hessian(x):
            hess = hessian(function)(x)
            if len(hess.shape) == 3:
                return hess[0][0]

            return hess

        self.hessian = fixed_hessian
        self.lim = lim
        self.point_min = np.array(point_min)


functions_with_one_min: list = [
    test_function(lambda x: (x - 5.0) ** 2.0 - 5.0, [-10, 10], [5.0]),
    test_function(lambda x: 10 * (x - math.sqrt(2)) ** 2 - math.sqrt(3), [-10, 10], [math.sqrt(2)]),
    test_function(lambda x: (x + 1.5) ** 2 / 10 + 9, [-10, 10], [-1.5]),
]

functions_with_local_min: list = [
    test_function(lambda x: -1 / ((x + 0.5) ** 2 + 1) + np.sin(x / 2) / 5 + np.cos(x) / 3, [-5, 5], [-0.6564301176463407]),
    test_function(lambda x: x ** 4 / 100.0 - x ** 3 / 10 - x ** 2 / 2 + 2 * x + 5, [-10, 15], [9.56688395672316]),
    test_function(lambda x: np.sin(4 * x) + x, [-4, 4],[ -3.597461800926115]),
    test_function(lambda x: -x * np.sin(-4 * x) - x, [-5, 5], [4.348404465645833]),
    test_function(lambda x: 2 ** (np.sin(-x)) + -x / 10, [-5, 5], [1.8552300957458492]),
    test_function(
        lambda x: -x * np.sin(-4 * x) - x + np.sin(15 * x) / 5 + np.cos(16 * x) / 3 + np.sin(20 * x) / 2 + np.cos(
            32 * x) / 16, [-5, 5], [4.356078353351506])
]

functions_with_one_min_2d: list = [
    test_function(lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2, [-100, 100], [0.5, 0.5]),
    test_function(lambda x: (x[0] - math.sqrt(2)) ** 2 + (x[1] + math.sqrt(3)) ** 2 + math.sqrt(5), [-100, 100], [math.sqrt(2), -math.sqrt(3)]),

    test_function(lambda x: ((x[0] + 0.5) ** 2) / 10 + ((x[1] - 1.5) ** 2) / 10 - 3, [-100, 100], [-0.5, 1.5]),
    test_function(lambda x: ((x[0] + 0.5) ** 2) / 100 + ((x[1] - 1.5) ** 2) / 100 - 3, [-100, 100], [-0.5, 1.5]),
    test_function(lambda x: 5 * ((x[0] - math.sqrt(9)) ** 2) + ((x[1] + math.sqrt(7)) ** 2) / 5 + math.sqrt(3), [-100, 100], [math.sqrt(9), -math.sqrt(7)]),
    test_function(lambda x: 1 / (x[0] ** 2 + x[1] ** 2 + 0.1) + x[0] ** 2 + x[1] ** 2 + x[1] / 10, [-5, 5], [0, -0.96282]),
    test_function(lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 10 / ((x[0] - 0.5) ** 2 + 1), [-5, 5], [0.5, 0.5])
]


functions_with_local_min_2d: list = [
    test_function(lambda x: 1 / (x[0] ** 2 + x[1] ** 2 + 0.1) + x[0] ** 2 + x[1] ** 2 + x[1] / 10 + 0.5 * (
            np.cos(10 * (x[1] ** 2 + x[0] ** 2) ** 0.5 - 2) + np.sin(16 * x[1] - 18) / 3 + np.sin(
        25 * x[0]) / 30 + np.sin(
        32 * x[0]) / 16 + np.sin(18 * x[0] + 8) / 4 + np.cos(29 * x[1] - 5) / 13 + x[0] * np.sin(x[1]) + np.sin(
        32 * x[0]) / 16 + np.sin(0.5 * x[0]) * 10 + 20 * np.sin(0.1 * (x[1] ** 2 + x[0] ** 2) ** 0.5) + np.sin(
        x[1] - 5) / (5 + x[0] ** 2)), [-3, 3], [-1.18833355,  0.24699018]),
    test_function(lambda x: x[0] ** 2 + x[1] ** 2 + 0.3 * (np.cos(10 * (x[1] ** 2 + x[0] ** 2) ** 0.5 - 2) + np.sin(16 * x[1] - 18) / 3 + np.sin(
        25 * x[0]) / 30 + np.sin(
        32 * x[0]) / 16 + np.sin(18 * x[0] + 8) / 4 + np.cos(29 * x[1] - 5) / 13 + x[0] * np.sin(x[1]) + np.sin(
        32 * x[0]) / 16 + np.sin(0.5 * x[0]) * 10 + 20 * np.sin(0.1 * (x[1] ** 2 + x[0] ** 2) ** 0.5) + np.sin(
        x[1] - 5) / (5 + x[0] ** 2)) ** 2, [-10, 10], [0,  0]),
    test_function(lambda x: -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2), [-100, 100], [np.pi, np.pi]),
    test_function(lambda x: np.sin(3 * np.pi * x[0]) ** 2 +(x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2) + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2), [-10, 10], [1, 1])
]
