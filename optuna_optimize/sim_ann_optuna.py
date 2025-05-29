import optuna
from functools import partial
import numpy as np
import optimize.simulated_annealing as sim_an
import common.tests_function as functions
import optimize.gradient_descent as gr
import optimize.tsp as tsp

cities = tsp.generate_points(n_cities=20, map_size=100)
dist_matrix = tsp.distance_matrix(cities)
f = tsp.create_route_len(dist_matrix)
start = tsp.create_route(20)

def objective(trial):

    t_min = trial.suggest_float("t_min", 1e-5, 1e-1)
    t_start = trial.suggest_float("t_start", 100, 10000)
    cooling_rate = trial.suggest_float("colling_rate", 0.90, 0.99)
    max_count = trial.suggest_int("max_count", 1000, 10000)
    res2 = sim_an.simulated_annealing(f, start, tsp.get_next_route, t_start, cooling_rate, t_min, max_count)

    return f(res2.get_res())


study = optuna.create_study()
study.optimize(objective, n_trials=1000)

print("Лучшие параметры:", study.best_params)
print("Лучшее значение функции:", study.best_value)