import optuna
from functools import partial
import numpy as np
import optimize.simulated_annealing as sim_an
import common.tests_function as functions
import optimize.gradient_descent as gr

numb_func = 1
test_func = functions.functions_with_local_min_2d[numb_func]

# Целевая функция для Optuna
def objective(trial):

    t_min = trial.suggest_float("t_min", 1e-5, 1e-1)
    t_start = trial.suggest_float("t_start", 100, 10000)
    cooling_rate = trial.suggest_float("colling_rate", 0.90, 0.99)
    max_count = trial.suggest_int("max_count", 1000, 10000)
    step_size = trial.suggest_int("step_size", 5, 100)

    lim = test_func.lim
    x = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    y = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    result = sim_an.simulated_annealing(
            test_func.function,
            [x, y],
            sim_an.get_next_in_sqrt((lim[1] - lim[0]) / step_size),
            t_start,
            cooling_rate,
            t_min,
            max_count
        )
    return (abs(test_func.function(result.get_res()) - test_func.function(test_func.point_min))) * result.count_of_function_calls


# Исследование
study = optuna.create_study()
study.optimize(objective, n_trials=2000)

# Результаты
print("Лучшие параметры:", study.best_params)
print("Лучшее значение функции:", study.best_value)