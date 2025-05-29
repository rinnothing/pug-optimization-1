import common.datasets
import optimize.sgd as sgd
import optimize.gradient_descent as gr

import numpy as np
from sklearn.model_selection import train_test_split

print("batch", "l_1","l_2", "moment", "deviation", "iter", "calls", "grad", sep=" & ", end=" \\\\ \\hline\n")

def launch(func, batch_size, l_1, l_2, moment):
    f = common.datasets.func_dataset[func]
    weights_0 = np.ones(3)

    # trying to align using second order curve
    a = lambda x, w: np.array([w[0] + x[0] * w[1] + x[0] ** 2 * w[2]])
    a_der = lambda x, w: np.array([1, x[0], x[0] ** 2])

    X_train, X_test, y_train, y_test = train_test_split(f.X, f.y, test_size=0.9, random_state=42)
    model = sgd.SGDLearn(weights_0, *sgd.regularize_elastic(*sgd.square_loss(a, a_der), l_1, l_2),
                         gr.get_stop_x_eps(0.001), gr.get_next_wolfe, batch_size)
    res = model.fit(X_train, y_train, min_count=300, max_count=1000, moment=moment)

    average_test = 0
    for point, val in zip(X_test, y_test):
        average_test += (a(point, model.weights) - val) ** 2
    average_test /= len(X_test)

    print(batch_size, l_1, l_2, moment, average_test[0], len(res.guesses),
          res.count_of_function_calls, res.count_of_gradient_calls, sep=" & ", end=" \\\\ \\hline\n")

moment_list = [0, 0.1, 0.5]

batch_list = [1, 5, 50]

l_1_list = [0.01, 0.5]

l_2_list = [0.01, 0.5]

func = 1
for batch in batch_list:
    for l_1 in l_1_list:
        for l_2 in l_2_list:
            for moment in moment_list:
                launch(func, batch, l_1, l_2, moment)
