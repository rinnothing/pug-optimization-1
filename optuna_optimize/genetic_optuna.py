import optuna
from functools import partial
import numpy as np
import optimize.genetic as gen
import common.tests_function as functions
import optimize.gradient_descent as gr
import optimize.tsp as tsp

numb_func = 1
test_func = functions.functions_with_local_min_2d[numb_func]
cities = tsp.generate_points(n_cities=20, map_size=100)
dist_matrix = tsp.distance_matrix(cities)
f = tsp.create_route_len(dist_matrix)
start = tsp.create_route(20)

def objective(trial):
    pop_size = trial.suggest_int("pop_size", 30, 100)
    generations = trial.suggest_int("generations", 40, 150)
    selection_count = trial.suggest_int("selection_count", 2, 20)
    crossover_rate = trial.suggest_float("crossover_rate", 0.6, 1)
    mutate_rate = trial.suggest_float("mutate_rate", 0.3, 0.8)

    lim = test_func.lim
    x = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    y = ((lim[1] - lim[0]) * 3) / 4  + lim[0]
    result = tsp.gen_tsp(
            f,
            20,
            pop_size,
            generations,
            selection_count,
            crossover_rate,
            mutate_rate
        )
    return f(result.get_res())


study = optuna.create_study()
study.optimize(objective, n_trials=1000)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение функции:", study.best_value)