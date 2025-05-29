import optuna
from functools import partial
import numpy as np
import optimize.bfgs as bfgs
import common.tests_function as functions
import optimize.gradient_descent as gr

numb_func = 5
test_func = functions.functions_with_one_min_2d[numb_func]

def objective(trial):

    f_eps = trial.suggest_float("f_eps", 1e-9, 1e-8)
    c_1 = trial.suggest_float("wolfe_c1", 0.001, 0.04)
    c_2 = trial.suggest_float("wolfe_c2", 0.83, 0.95)
    max_count = trial.suggest_int("max_count", 1000, 10000)
    max_count_wolfe = trial.suggest_int("max_count_wolfe", 20, 100)

    lim = test_func.lim
    x = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    y = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    result = bfgs.bfgs(test_func.function, test_func.gradient,
                  partial(gr.get_next_wolfe, c1=c_1, c2=c_2, max_count = max_count_wolfe),
                  gr.get_stop_x_eps(f_eps), np.array([x, y]), max_count = max_count)
    return abs(test_func.function(result.get_res()) - test_func.function(test_func.point_min)) * (result.count_of_function_calls + result.count_of_gradient_calls + result.count_of_hessian_calls)


study = optuna.create_study()
study.optimize(objective, n_trials=200)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение функции:", study.best_value)