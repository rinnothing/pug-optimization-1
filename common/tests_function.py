from autograd import grad

import numpy as np

class test_function:
    def __init__(self, function, lim):
        self.function = function
        self.gradient = grad(function)
        self.lim = lim


functions_with_one_min: list = [
    test_function(lambda x: (x - 5.0) ** 2.0 - 5.0, [-5, 10]),
    test_function(lambda x: np.sin(x)/(x ** 2 + 1), [2, 7.5]),
    test_function(lambda x: 1/(x ** 2 + 1), [0, 5]),
    test_function(lambda x: 1/(x ** 2 + 1), [5, 10]),
    test_function(lambda x: -1/((x + 0.5) ** 2 + 1), [-5, 5])
]

functions_with_local_min:list = [
    test_function(lambda x: -1/((x + 0.5) ** 2 + 1) + np.sin(x/2)/5 + np.cos(x)/3, [-5, 5]),
    test_function(lambda x:  x ** 4 / 100.0 - x ** 3 / 10 - x ** 2 / 2 + 2 * x + 5, [-10, 15]),
    test_function(lambda x: np.sin(4*x) + x, [-4, 4]),
    test_function(lambda x: -x * np.sin(-4*x) - x, [-5, 5]),
    test_function(lambda x: 2 ** (np.sin(-x)) + -x/10, [-5, 5]),
]
