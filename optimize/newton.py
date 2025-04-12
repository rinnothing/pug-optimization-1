import common
import common.tests_function
from gradient_descent import create_grad_from_bad_func, symmetric_derivative, get_next_wolfe, get_stop_x_eps, get_constant_step
import optimize.general_visualiser as vis

import numpy as np


def newton(fun, grad, hess, get_next, trust, stop, x, min_x=np.array([-100, -100]), max_x=np.array([100, 100]), min_count=10,
           max_count=100):
    res = common.StateResult()
    res.function = fun

    res.add_guess(x)
    save_min_value = [x, fun(x)]

    count = 0
    while (not stop(res) or min_count > count) and max_count > count:
        count += 1

        grad_val = grad(x)
        hess_val = hess(x)
        if len(hess_val.shape) != 1:
            inv_hess_val = np.linalg.inv(hess_val)
        else:
            inv_hess_val = np.array([1.0 / hess_val[0]])
        base_p = -inv_hess_val * grad_val

        to_optimize = lambda p: (grad_val * base_p) * p + (base_p / 2 * hess_val * base_p) * p * p
        local_grad = lambda p: grad_val + (hess_val * base_p) * p
        local_antigrad = -local_grad(0)
        new_p = get_next(res, to_optimize, local_grad, local_antigrad, 0) * base_p

        trust_val = trust(res)
        if trust_val is not None:
            new_p = max(new_p, -trust_val)
            new_p = min(new_p, trust_val)
        x = x + base_p * new_p

        fun_x = fun(x)
        if fun_x < save_min_value[1]:
            save_min_value = [x, fun_x]
        for i in range(len(x)):
            x[i] = min(x[i], max_x[i])
            x[i] = max(x[i], min_x[i])

        res.add_guess(x)

    res.add_guess(save_min_value[0])
    res.success = True
    return res


def get_real_const_step(h0):
    def get_next(state, fun, grad, antigrad_val, x):
        return x + h0

    return get_next

def get_const_trust(tr0):
    def get_trust(state):
        return tr0

    return get_trust

def no_trust(state):
    return None

# a simple way to implement hess
def create_hess_from_bad_func(fun, bad_func, *args, **kwargs):
    der = lambda x: bad_func(fun, x, *args, **kwargs)
    def grad(x):
        return bad_func(der, x, *args, **kwargs)

    return grad


if __name__ == "__main__":
    for test_func in common.tests_function.functions_with_one_min:
        lim = test_func.lim
        result = newton(test_func.function, create_grad_from_bad_func(test_func.function, symmetric_derivative, delta=0.1),
                        create_hess_from_bad_func(test_func.function, symmetric_derivative, delta=0.1), get_real_const_step(1),
                        get_const_trust(0.5), get_stop_x_eps(0.01), np.array([lim[0]]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        new_lim = np.array([lim[0] - lim[1] + lim[0], lim[1]])
        new_lim[0] = min(new_lim[0], result.get_res() - 1)
        new_lim[1] = max(new_lim[1], result.get_res() + 1)
        vis.visualiser(result, new_lim, 500)
    print("start functions with local min")
    for test_func in common.tests_function.functions_with_local_min:
        lim = test_func.lim
        result = newton(test_func.function, test_func.gradient,
                        create_grad_from_bad_func(test_func.gradient, symmetric_derivative), get_real_const_step(1),
                        get_const_trust(0.5), get_stop_x_eps(0.01), np.array([lim[0]]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        if not result.success:
            print("didn't solve")
        new_lim = np.array([lim[0] - lim[1] + lim[0], lim[1]])
        new_lim[0] = min(new_lim[0], result.get_res() - 1)
        new_lim[1] = max(new_lim[1], result.get_res() + 1)
        vis.visualiser(result, new_lim, 500)
