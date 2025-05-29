import optuna
from functools import partial
import numpy as np
import optimize.newton as nt
import common.tests_function as functions
import optimize.gradient_descent as gr

numb_func = 5
test_func = functions.functions_with_one_min_2d[numb_func]

def objective(trial):

    f_eps = trial.suggest_float("f_eps", 1e-9, 1e-8)
    c_1 = trial.suggest_float("wolfe_c1", 0.43, 0.6)
    c_2 = trial.suggest_float("wolfe_c2", 0.87, 0.96)
    max_count = trial.suggest_int("max_count", 1000, 10000)

    lim = test_func.lim
    x = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    y = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    result = nt.newton(test_func.function, test_func.gradient, test_func.hessian, partial(gr.get_next_wolfe, c1=c_1, c2=c_2, max_count = 10), nt.get_const_trust(None),gr.get_stop_x_eps(f_eps), np.array([x, y]), max_count = max_count)
    return abs(test_func.function(result.get_res()) - test_func.function(test_func.point_min)) * (result.count_of_function_calls + result.count_of_gradient_calls + result.count_of_hessian_calls)


study = optuna.create_study()
study.optimize(objective, n_trials=100)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение функции:", study.best_value)