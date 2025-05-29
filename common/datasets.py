import random

import pandas
import numpy as np

import optimize.sgd as sgd


class dataset:
    def __init__(self, x, y):
        self.X = x
        self.y = y

def fun_with_error(x, fun, eps):
    random.seed(42)
    return dataset(x, [fun(x_small) + random.random() * eps for x_small in x])

electricity_dataset = pandags.read_csv("../dataset/electricity/ex_1.csv")
actual: list = [
    dataset(electricity_dataset[['time', 'input_voltage']].to_numpy(), electricity_dataset['el_power'].to_numpy())
]

func_dataset: list = [
    fun_with_error([[x] for x in range(1, 1000)], lambda x: 5 * x[0] + 1, 1),
    fun_with_error([[x] for x in range(1, 1000)], lambda x: 35 * x[0] ** 2 - 10 * x[0] + 5, 5),
]
