import numpy as np
import common
import random
import common.tests_function
import autograd.numpy as anp  # Автоматическое дифференцирование
from scipy.optimize import line_search
import optimize.general_visualiser as vis
import optimize.gradient_descent as gr
import optimize.multidim_gradient as mul_gr


def bfgs(fun, grad, get_next, stop, x, max_count=100, min_count = 10, tol=1e-6):

    res = common.StateResult()
    res.function = fun
    res.add_guess(x)

    n = len(x)
    H = np.eye(n)

    count = 0
    while (not stop(res) or min_count > count) and max_count > count:
        count += 1

        g_val = grad(x)
        res.add_gradient_call()
        pk = -H.dot(g_val)

        preres = get_next(res, fun, grad, pk, x)
        res.count_of_function_calls += preres.count_call_func
        res.count_of_gradient_calls += preres.count_call_grad
        res.count_of_hessian_calls += preres.count_call_hess
        x_new = preres.res
        s = x_new - x
        y = grad(x_new) - g_val
        res.add_gradient_call()

        if np.dot(y, s) <= 1e-10:
            x = x_new
            res.add_guess(x)
            continue

        rho = 1.0 / np.dot(y, s)
        A1 = np.eye(n) - rho * np.outer(s, y)
        A2 = np.eye(n) - rho * np.outer(y, s)
        H = A1.dot(H.dot(A2)) + rho * np.outer(s, s)

        x = x_new
        res.add_guess(x)
    res.success = True
    return res


if __name__ == "__main__":
    for test_func in common.tests_function.functions_with_one_min:
        lim = test_func.lim
        result = bfgs(test_func.function, test_func.gradient,
                      gr.get_next_wolfe,
                      gr.get_stop_f_eps(0.01), np.array([lim[0]]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        print("Count of gradient calls: ", result.count_of_gradient_calls, "Count of hessian calls: ",
              result.count_of_hessian_calls)
        if not result.success:
            print("didn't solve")
        new_lim = np.array([lim[0] - lim[1] + lim[0], lim[1]])
        new_lim[0] = min(new_lim[0], result.get_res() - 1)
        new_lim[1] = max(new_lim[1], result.get_res() + 1)
        vis.visualiser(result, new_lim, 500)
    print("start functions with local min")
    for test_func in common.tests_function.functions_with_local_min:
        lim = test_func.lim

        result = bfgs(test_func.function, test_func.gradient,
                      gr.get_next_wolfe,
                      gr.get_stop_f_eps(0.01), np.array([lim[0]]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        print("Count of gradient calls: ", result.count_of_gradient_calls, "Count of hessian calls: ",
              result.count_of_hessian_calls)
        if not result.success:
            print("didn't solve")
        new_lim = np.array([lim[0] - lim[1] + lim[0], lim[1]])
        new_lim[0] = min(new_lim[0], result.get_res() - 1)
        new_lim[1] = max(new_lim[1], result.get_res() + 1)
        vis.visualiser(result, new_lim, 500)

    for test_func in common.tests_function.functions_with_one_min_2d:
        lim = test_func.lim
        x = random.uniform(lim[0], lim[1])
        y = random.uniform(lim[0], lim[1])
        result = bfgs(test_func.function, test_func.gradient,
                        gr.get_next_wolfe,
                      gr.get_stop_f_eps(0.01), np.array([x, y]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        print("Count of gradient calls: ", result.count_of_gradient_calls, "Count of hessian calls: ",
              result.count_of_hessian_calls)
        if not result.success:
            print("didn't solve")
        else:
            mul_gr.visualiser_2d(result, lim_x=lim, lim_y=lim)
            mul_gr.visualiser_3d(result, lim_x=lim, lim_y=lim)
