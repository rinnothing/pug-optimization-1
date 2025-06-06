from memory_profiler import profile
import random

import numpy as np
import pandas
import sklearn.preprocessing
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

    @profile
    def fit(self, x, y, min_count=1, max_count=100, moment=0.0):
        res = common.StateResult()
        if len(res.guesses) == 0:
            res.add_guess(self.weights)

        count = 0
        vec=np.zeros_like(self.weights)

        while (not self.stop(res) or min_count > count) and max_count > count:
            if self.batch != 0:
                ran = get_n_random(self.batch, len(x))
                size = self.batch
            else:
                ran = range(0, len(x))
                size = len(x)

            grad = 0
            for n in ran:
                # just like a normal derivative
                grad += self.loss_der(self.weights, x[n], y[n])
            grad /= size

            def optim_fun(val):
                # I don't know, maybe it's better to call it for every value
                ans = 0
                for n in ran:
                    ans += self.loss(val, x[n], y[n])
                ans /= size

                return ans

            def optim_der(val):
                # the same as upper
                ans = 0
                for n in ran:
                    ans += self.loss_der(val, x[n], y[n])
                ans /= size

                return ans

            vec = moment * vec + (1 - moment) * grad
            preres = self.get_next(res, optim_fun, optim_der, -vec, self.weights)
            res.count_of_function_calls += preres.count_call_func * size
            res.count_of_gradient_calls += preres.count_call_grad * size + size
            res.count_of_hessian_calls += preres.count_call_hess * size

            self.weights = preres.res
            res.add_guess(self.weights)
            count += 1

        res.add_guess(self.weights)
        res.success = True
        return res


def get_n_random(n, sz):
    if n > sz:
        return

    already_used = dict()
    while len(already_used) < n:
        gen = random.randint(0, sz-1)
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
            loss_val += l1 * np.sum(np.abs(weights))

        if l2 != 0:
            loss_val += l2 * np.sum(np.square(weights))

        return loss_val

    def local_loss_der(weights, x, y):
        loss_val = loss_der(weights, x, y)
        if l1 != 0:
            arr = weights.copy()
            arr[arr > 0] = 1
            arr[arr < 0] = -1

            loss_val += l1 * np.sum(arr)

        if l2 != 0:
            loss_val += l2 * np.sum(weights) * 2

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

def normalize(arr):
    return np.divide(arr, np.max(arr))

def denormalize(weights, x, y):
    return np.multiply(weights, np.max(y) / np.max(x))

if __name__ == "__main__":
    # # fetch dataset
    dataset = pandas.read_csv("../dataset/electricity/ex_1.csv")
    #
    # # data (as pandas dataframes)
    X = dataset[['time', 'input_voltage']].to_numpy()
    y = dataset['el_power'].to_numpy()

    X_norm = normalize(X)
    y_norm = normalize(y)

    # just some initial weights (better use random, but I didn't find the way)
    weights_0 = np.ones(6)

    # trying to align using second order curve
    a = lambda x, w: np.array([w[0] + x[0] * w[1] + x[0] ** 2 * w[2] + w[3] + x[1] * w[4] + x[1] ** 2 * w[5]])
    a_der = lambda x, w: np.array([1, x[0], x[0] ** 2, 1, x[0], x[0] ** 2])

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.9, random_state=42)
    model = SGDLearn(weights_0, *regularize_elastic(*square_loss(a, a_der), 0.01, 0.01),
                     gr.get_stop_x_eps(0.001), gr.get_next_wolfe, 30)
    res = model.fit(X_train, y_train, min_count=300, max_count=1000, moment=0)

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
