import common
import optimize.gradient_descent as gr

from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import common.tests_function as test_f


# if you know derivative explicitly you can use it, but in other cases take a look at those functions
def a_b_derivative(fun, x, a, b):
    matrix_x = np.broadcast_to(x, (x.size, x.size))

    matrix_xa = np.zeros(matrix_x.shape)
    np.fill_diagonal(matrix_xa, a)
    matrix_xa += matrix_x

    matrix_xb = np.zeros(matrix_x.shape)
    np.fill_diagonal(matrix_xb, -b)
    matrix_xb += matrix_x

    return (np.apply_along_axis(fun, 0, matrix_xa) - np.apply_along_axis(fun, 0, matrix_xb)) / (a + b)


def forward_derivative(fun, x, delta=1.0):
    """
    calculates forward derivative
    :param fun: function to calculate derivative
    :param x: point in which to calculate
    :param delta: which delta to use (1.0 by default)
    :return: derivative
    """
    return a_b_derivative(fun, x, a=delta, b=0)


def backward_derivative(fun, x, delta=1.0):
    """
    calculates backward derivative
    :param fun: function to calculate derivative
    :param x: point in which to calculate
    :param delta: which delta to use (1.0 by default)
    :return: derivative
    """
    return a_b_derivative(fun, x, a=0, b=delta)


def symmetric_derivative(fun, x, delta=1.0):
    """
    calculates symmetric derivative
    :param fun: function to calculate derivative
    :param x: point in which to calculate
    :param delta: which delta to use (1.0 by default)
    :return: derivative
    """
    return a_b_derivative(fun, x, delta, delta)


# visualiser

def visualiser_2d(state: common.StateResult, lim_x, lim_y):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z = state.function([X, Y])

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    x = state.guesses[0]
    arrow = matplotlib.patches.Arrow(x[0], x[1], state.guesses[0][0] - x[0], state.guesses[0][1] - x[1], width=0.5,
                                     color="red")
    ax.add_patch(arrow)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def animate(i):
        arrow.set_data(state.guesses[i][0], state.guesses[i][1],
                       state.guesses[i + 1][0] - state.guesses[i][0], state.guesses[i + 1][1] - state.guesses[i][1])
        return arrow

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(state.guesses) - 1,
        repeat=True,
        interval=100
    )
    plt.show()


# maybe will add other later
def visualiser_3d(state: common.StateResult, lim_x, lim_y):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 100), np.linspace(lim_y[0], lim_y[1], 100))
    Z = state.function([X, Y])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    guesses = np.array(state.guesses)
    ax.plot(guesses[:, 0], guesses[:, 1], state.function(guesses.T), 'ro-', markersize=5)

    point, = ax.plot([], [], [], 'bo', markersize=8)

    def animate(i):
        point.set_data_3d([guesses[i, 0]], [guesses[i, 1]], [state.function(guesses[i])])
        return point,

    ani = animation.FuncAnimation(fig, animate, frames=len(state.guesses), interval=100, blit=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    plt.show()

if __name__ == "__main__":
    for test_func in test_f.functions_with_one_min_2d:
        lim = test_func.lim
        result = gr.gradient_descent(test_func.function, test_func.gradient, gr.get_next_wolfe,
                                     gr.get_stop_f_eps(0.0001),
                                     np.array([-1.9, 1.5]))
        if not result.success:
            print("I'm sorry, no solution")
        else:
            print(result.get_res(), len(result.guesses))
            print(result.guesses)
            print(result.history)
            visualiser_2d(result, lim_x=lim, lim_y=lim)
            visualiser_3d(result, lim_x=lim, lim_y=lim)
