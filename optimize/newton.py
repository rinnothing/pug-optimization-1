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

        to_optimize = lambda p: np.dot(grad_val, base_p) * p + np.dot(
            np.dot(base_p, hess_val), base_p) / 2 * p * p
        local_grad = lambda p: np.dot(grad_val, base_p) + np.dot(
            np.dot(base_p, hess_val), base_p) * p
        local_antigrad = -local_grad(0)
        preres = get_next(res, to_optimize, local_grad, local_antigrad, 0)
        res.count_of_function_calls += preres.count_call_func
        res.count_of_gradient_calls += preres.count_call_grad
        res.count_of_hessian_calls += preres.count_call_hess
        new_p = preres.res * base_p

        trust_val = trust(res)
        if trust_val is not None:
            new_p = min(norm(new_p), trust_val)
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
    for test_func in common.tests_function.functions_with_one_min:
        lim = test_func.lim
        result = newton(test_func.function, test_func.gradient,
                        test_func.hessian, gr.get_next_wolfe,
                        get_const_trust(0.5), gr.get_stop_f_eps(0.01), np.array([lim[0]]))
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
        result = newton(test_func.function, test_func.gradient,
                        test_func.hessian, gr.get_next_wolfe,
                        get_const_trust(0.5), gr.get_stop_f_eps(0.01), np.array([lim[0]]))
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
        result = newton(test_func.function, test_func.gradient,
                        test_func.hessian, gr.get_next_wolfe,
                        get_const_trust(0.5), gr.get_stop_f_eps(0.01), np.array([x, y]))
        print("Count of function calls: ", result.count_of_function_calls, " | result: ", result.get_res())
        print("Count of gradient calls: ", result.count_of_gradient_calls, "Count of hessian calls: ",
              result.count_of_hessian_calls)
        if not result.success:
            print("didn't solve")
        else:
            mul_gr.visualiser_2d(result, lim_x=lim, lim_y=lim)
            mul_gr.visualiser_3d(result, lim_x=lim, lim_y=lim)
