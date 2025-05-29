from typing import Protocol

from scipy.optimize import OptimizeResult


def zero(x):
    return 0

class res_and_count:
    def __init__(self, res, count_call_func, count_call_grad, count_call_hess = 0):
        self.res = res
        self.count_call_func = count_call_func
        self.count_call_grad = count_call_grad
        self.count_call_hess = count_call_hess

class StateResult(OptimizeResult):
    """
    Represents result and internal state of optimization
    Should be exactly like `OptimizeResult`, but also store information on previous guesses

    Attributes
    ----------
    guesses vals you guessed on every step
    history some information on bounds and other stuff you need after run
    function is a function on which it did optimizations
    count_of_function_calls is count of function calls
    count_of_gradient_calls is count of gradient calls
    """

    def __init__(self):
        super().__init__()

        self.success: bool = False
        self.guesses: list = []
        self.new_guesses: list = []
        self.history: list = []
        self.count_of_function_calls: int = 0
        self.count_of_gradient_calls: int = 0
        self.count_of_hessian_calls: int = 0

    def add_guess(self, guess):
        self.guesses.append(guess)

    def add_history(self, val):
        self.history.append(val)

    def add_function_call(self, count = 1):
        self.count_of_function_calls += count

    def add_gradient_call(self, count = 1):
        self.count_of_gradient_calls += count

    def add_hessian_call(self, count = 1):
        self.count_of_hessian_calls += count

    def get_res(self):
        if not self.success or len(self.guesses) == 0:
            return None

        return self.guesses[-1]

    # to be continued


def get_times_stopper(stopper, times):
    """
    gives given stopper a run times restriction
    :param stopper: stopper function
    :param times: max number of times it should be run
    :return: stopper with applied restrictions
    """
    k = 0

    def times_stopper(state):
        nonlocal k

        if k >= times:
            return False

        k += 1

        return stopper(state)

    return times_stopper
