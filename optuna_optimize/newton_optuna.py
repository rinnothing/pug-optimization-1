from functools import partial

import optimize.newton as newton
import common
import common.tests_function
import optimize.gradient_descent as gr

import optuna
import numpy as np

import random

def to_optimize(trial):
    val_sum = 0

    trust = trial.suggest_float("trust_period", 0, 100)
    f_eps = trial.suggest_float("f_eps", 0, 0.1)
    c_1 = trial.suggest_float("wolfe_c1", 0, 0.05)
    c_2 = trial.suggest_float("wolfe_c2", 0, 0.8)

    for test_func in common.tests_function.functions_with_one_min:
        lim = test_func.lim
        result = newton.newton(test_func.function, test_func.gradient,
                        test_func.hessian, partial(gr.get_next_wolfe, c1=c_1, c2=c_2),
                        newton.get_const_trust(trust), gr.get_stop_f_eps(f_eps), np.array([lim[0]]))
        val_sum += test_func.function(result.get_res())

    for test_func in common.tests_function.functions_with_local_min:
        lim = test_func.lim
        result = newton.newton(test_func.function, test_func.gradient,
                        test_func.hessian, partial(gr.get_next_wolfe, c1=c_1, c2=c_2),
                        newton.get_const_trust(trust), gr.get_stop_f_eps(f_eps), np.array([lim[0]]))
        val_sum += test_func.function(result.get_res())

    for test_func in common.tests_function.functions_with_one_min_2d:
        lim = test_func.lim
        x = random.uniform(lim[0], lim[1])
        y = random.uniform(lim[0], lim[1])
        result = newton.newton(test_func.function, test_func.gradient,
                        test_func.hessian, partial(gr.get_next_wolfe, c1=c_1, c2=c_2),
                        newton.get_const_trust(trust), gr.get_stop_f_eps(f_eps), np.array([x, y]))
        val_sum += test_func.function(result.get_res())

    # just returning the sum of all answers
    return val_sum

if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(to_optimize, n_trials=100)

    best_params = study.best_params
    print("best params: ", best_params)
