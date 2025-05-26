import random

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

import common
import optimize.gradient_descent as gr


class SGDLearn:
    def __init__(self, w_0, loss, loss_der, stop, get_next, batch=0):
        self.loss = loss
        self.loss_der = loss_der
        self.stop = stop
        self.get_next = get_next

        self.weights = w_0
        self.batch = batch

    def fit(self, x, y, min_count=1, max_count=100):
        res = common.StateResult()
        if len(res.guesses) == 0:
            res.add_guess(self.weights)

        count = 0
        while (not self.stop(res) or min_count > count) and max_count > count:
            if self.batch != 0:
                ran = get_n_random(self.batch, len(x))
            else:
                ran = range(0, self.batch)

            grad = 0
            for n in ran:
                # just like a normal derivative
                grad += self.loss_der(self.weights, x[n], y[n])

            def optim_fun(val):
                # I don't know, maybe it's better to call it for every value
                ans = 0
                for n in ran:
                    ans += self.loss(self.weights, x[n], y[n])

                return ans

            def optim_der(val):
                # the same us upper
                ans = 0
                for n in ran:
                    ans += self.loss_der(self.weights, x[n], y[n])

                return ans

            preres = self.get_next(res, optim_fun, optim_der, -grad, self.weights)
            res.count_of_function_calls += preres.count_call_func
            res.count_of_gradient_calls += preres.count_call_grad
            res.count_of_hessian_calls += preres.count_call_hess

            self.weights = preres.res
            res.add_guess(self.weights)

        res.add_guess(self.weights)
        res.success = True
        return res


def get_n_random(n, sz):
    if n > sz:
        return

    already_used = dict()
    while len(already_used) < n:
        gen = random.randint(0, n)
        if gen not in already_used:
            already_used[gen] = True

    return already_used.keys()


def regularize_l1(loss, loss_der, l1):
    return regularize_elastic(loss, loss_der, l1, 0)


def regularize_l2(loss, loss_der, l2):
    return regularize_elastic(loss, loss_der, 0, l2)


def regularize_elastic(loss, loss_der, l1, l2):
    def local_loss_fun(weights, x, y):
        loss_val = loss(weights, x, y)
        if l1 != 0:
            loss_val += l1 * np.abs(weights)

        if l2 != 0:
            loss_val += l2 * np.square(weights)

        return loss_val

    def local_loss_der(weights, x, y):
        loss_val = loss_der(weights, x, y)
        if l1 != 0:
            arr = weights.copy()
            arr[arr > 0] = 1
            arr[arr < 0] = -1

            loss_val += l1 * arr

        if l2 != 0:
            loss_val += l2 * weights * 2

        return loss_val

    return local_loss_fun, local_loss_der


def square_loss(a, a_der):
    def local_loss_fun(weights, x, y):
        return np.square(a(x, weights) - y)

    def local_loss_der(weights, x, y):
        return 2 * (a(x, weights) - y) * a_der(x, weights)

    return local_loss_fun, local_loss_der


# I don't know why I implemented this
def sigmoid_loss(a, a_der):
    def local_loss_fun(weights, x, y):
        val = 1 / (1 + np.exp(-a(x, weights)))

        if y == 1:
            return val
        return 1 - val

    def local_loss_der(weights, x, y):
        eps = np.exp(-a(x, weights))

        val = -eps * a_der(x, weights) / np.square(1 + eps)

        if y == 1:
            return val
        return -val

    return local_loss_fun, local_loss_der


if __name__ == "__main__":
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # just some initial weights (better use random, but I didn't find the way)
    weights_0 = np.ones_like(wine_quality.data.features[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = SGDLearn(weights_0, *regularize_elastic(*square_loss(lambda x, w: np.dot(x, w), lambda x, w: w), 0.1, 0.2),
                     gr.get_stop_f_eps(0.1), gr.get_next_wolfe, 10)
