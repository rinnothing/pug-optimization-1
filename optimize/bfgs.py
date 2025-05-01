import numpy as np
from functools import partial
import common
import random
import common.tests_function
from scipy.optimize import line_search
import optimize.general_visualiser as vis
import optimize.gradient_descent as gr
import optimize.multidim_gradient as mul_gr


def bfgs(fun, grad, get_next, stop, x, max_count=6000, min_count = 1):

    res = common.StateResult()
    res.function = fun
    res.add_guess(x)

    n = len(x)
    H = np.eye(n)

    count = 0
    grad_new = grad(x)
    while (not stop(res) or min_count > count) and max_count > count:
        count += 1

        g_val = grad_new
        pk = -H.dot(g_val)

        preres = get_next(res, fun, grad, pk, x)
        res.count_of_function_calls += preres.count_call_func
        res.count_of_gradient_calls += preres.count_call_grad
        res.count_of_hessian_calls += preres.count_call_hess
        x_new = preres.res
        s = x_new - x
        grad_new = grad(x_new)
        y = grad_new - g_val
        res.add_gradient_call()

        if np.dot(y, s) <= 1e-10:
            x = x_new
            res.add_guess(x)
            continue

        p = 1.0 / np.dot(y, s)
        L = np.eye(n) - p * np.outer(s, y)
        R = np.eye(n) - p * np.outer(y, s)
        H = L.dot(H.dot(R)) + p * np.outer(s, s)

        x = x_new
        res.add_guess(x)
    res.success = True
    return res


if __name__ == "__main__":

    for test_func in common.tests_function.functions_with_one_min_2d:
        lim = test_func.lim
        x = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
        y = ((lim[1] - lim[0]) * 3) / 4 + lim[0]
        result = bfgs(test_func.function, test_func.gradient,
                        partial(gr.get_next_wolfe, c1=0.025, c2=0.84, max_count = 50),
                      gr.get_stop_f_eps(3e-5), np.array([x, y]), max_count = 10000)
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        print("Count of gradient calls: ", result.count_of_gradient_calls, "Count of hessian calls: ",
              result.count_of_hessian_calls)
        if not result.success:
            print("didn't solve")
        else:
            mul_gr.visualiser_2d(result, lim_x=lim, lim_y=lim)
            mul_gr.visualiser_3d(result, lim_x=lim, lim_y=lim)
