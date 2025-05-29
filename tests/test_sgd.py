import common.datasets
import optimize.sgd as sgd

import numpy as np
from sklearn.model_selection import train_test_split

for f in [common.datasets.func_dataset[0]]:
    weights_0 = np.ones(3)

    # trying to align using second order curve
    a = lambda x, w: np.array([w[0] + x[0] * w[1] + x[0] ** 2 * w[2]])
    a_der = lambda x, w: np.array([1, x[0], x[0] ** 2])

    X_train, X_test, y_train, y_test = train_test_split(f.X, f.y, test_size=0.9, random_state=42)
    model = sgd.SGDLearn(weights_0, *sgd.regularize_elastic(*sgd.square_loss(a, a_der), 0.007716671884570922, 0.0029403760996905256),
                         sgd.gr.get_stop_x_eps(0.001), sgd.gr.get_next_wolfe, 5)
    res = model.fit(X_train, y_train, min_count=131, max_count=1000)

    average_train = 0
    for point, val in zip(X_train, y_train):
        average_train += (a(point, model.weights) - val) ** 2
    average_train /= len(X_train)
    print("average train deviation: %s" % average_train)

    average_test = 0
    for point, val in zip(X_test, y_test):
        average_test += (a(point, model.weights) - val) ** 2
    average_test /= len(X_test)
    print("average test deviation: %s" % average_test)

    print("weights:", model.weights)
    print("number of iterations:", len(res.guesses))
    print("number of function calls", res.count_of_function_calls,
          "\nnumber of gradient calls", res.count_of_gradient_calls)
