import optuna
import numpy as np
import optimize.sgd as sgd
import common.datasets as datasets
import optimize.gradient_descent as gr
from functools import partial

from sklearn.model_selection import train_test_split

numb_func = 1
test_func = datasets.func_dataset[numb_func]

# Целевая функция для Optuna
def objective(trial):
    weights_0 = np.ones(3)

    l_1 = trial.suggest_float("l_1", 0.00001, 0.01)
    l_2 = trial.suggest_float("l_2", 0.00001, 0.01)
    batch = trial.suggest_int("batch", 1, 10)
    min_count = trial.suggest_int("min_count", 100, 1000)

    a = lambda x, w: np.array([w[0] + x[0] * w[1] + x[0] ** 2 * w[2]])
    a_der = lambda x, w: np.array([1, x[0], x[0] ** 2])

    X_train, X_test, y_train, y_test = train_test_split(test_func.X, test_func.y, test_size=0.9, random_state=42)
    model = sgd.SGDLearn(weights_0, *sgd.regularize_elastic(*sgd.square_loss(a, a_der), l_1, l_2),
                         sgd.gr.get_stop_x_eps(0.001), sgd.gr.get_next_wolfe, batch)
    result = model.fit(X_train, y_train, min_count=min_count, max_count=1000)

    average_test = 0
    for point, val in zip(X_test, y_test):
        average_test += (a(point, model.weights) - val) ** 2
    average_test /= len(X_test)

    return abs(average_test * (result.count_of_function_calls + result.count_of_gradient_calls + result.count_of_hessian_calls))


# Исследование
study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Результаты
print("Лучшие параметры:", study.best_params)
print("Лучшее значение функции:", study.best_value)