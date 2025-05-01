import random
import common
import common.tests_function
import optimize.general_visualiser as vis
import optimize.gradient_descent as gr
import optimize.multidim_gradient as mul_gr

import numpy as np
from numpy.linalg import norm


def newton(fun, grad, hess, get_next, trust, stop, x, min_x=np.array([-100, -100]), max_x=np.array([100, 100]),
           min_count=10,
           max_count=100):
    res = common.StateResult()
    res.function = fun

    res.add_guess(x)
    save_min_value = [x, fun(x)]

    count = 0
    while (not stop(res) or min_count > count) and max_count > count:
        count += 1

        grad_val = grad(x)
        res.add_gradient_call()
        hess_val = hess(x)
        res.add_hessian_call()
        if len(hess_val.shape) != 1:
            inv_hess_val = np.linalg.inv(hess_val)
        else:
            inv_hess_val = np.array([1.0 / hess_val[0]])
        base_p = np.dot(-inv_hess_val, np.matrix.transpose(grad_val))

        preres = get_next(res, fun, grad, base_p, x)
        res.count_of_function_calls += preres.count_call_func
        res.count_of_gradient_calls += preres.count_call_grad
        res.count_of_hessian_calls += preres.count_call_hess

        p = preres.res - x

        trust_val = trust(res)
        if trust_val is not None and norm(p) != 0:
            length = norm(p)
            p = min(length, trust_val) * (p / length)
        x = x + p

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
        return common.res_and_count(x + h0, 0, 0)

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
    for test_func in common.tests_function.functions_with_one_min_2d:
        lim = test_func.lim
        x = random.uniform(lim[0], lim[1])
        y = random.uniform(lim[0], lim[1])
        result = newton(test_func.function, test_func.gradient,
                        test_func.hessian, gr.get_next_wolfe,
                        get_const_trust(None), gr.get_stop_f_eps(1e-6), np.array([x, y]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        print("Count of gradient calls: ", result.count_of_gradient_calls, "Count of hessian calls: ",
              result.count_of_hessian_calls)
        if not result.success:
            print("didn't solve")
        else:
            mul_gr.visualiser_2d(result, lim_x=lim, lim_y=lim)
            mul_gr.visualiser_3d(result, lim_x=lim, lim_y=lim)
