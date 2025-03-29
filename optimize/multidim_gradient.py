import common
import gradient_descent as gr

from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# if you know derivative explicitly you can use it, but in other cases take a look at those functions
def a_b_derivative(fun, x, a, b):
    matrix_x = np.broadcast_to(x, (x.size, x.size))

    matrix_xa = np.zeros(matrix_x.shape)
    np.fill_diagonal(matrix_xa, a)
    matrix_xa += matrix_x

    matrix_xb = np.zeros(matrix_x.shape)
    np.fill_diagonal(matrix_xb, -b)
    matrix_xb += matrix_x

    return (np.apply_along_axis(fun, 1, matrix_xa) - np.apply_along_axis(fun, 1, matrix_xb)) / (a + b)


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

def gradient_visualiser(state: common.StateResult, limits, freq=50, l=1, interval=100, y_limits=None, path: str = None,
                        display=True):
    fig, ax = plt.subplots()

    # setting limits
    ax.set_xlim(limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    t = np.arange(limits[0], limits[1], (limits[1] - limits[0]) / freq)

    # drawing the base graphic
    ax.plot(t, state.function(t))

    # adding moving objects
    point, = ax.plot(0, 0, 'ro')
    arrow = ax.arrow(0, 0, 0, 0, )

    # making animation function
    def animate(i):
        nonlocal arrow
        arrow.remove()

        point_xy = (state.guesses[i], state.function(state.guesses[i]))
        point.set_data([[i] for i in point_xy])

        der, h = state.history[i]
        arrow = ax.arrow(point_xy[0], point_xy[1], -der * h * l, -der * der * h * l)

        return point, arrow

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(state.history),
        repeat=True,
        interval=interval
    )

    if path is not None:
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(path, writer=writer)

    if display:
        plt.show()


# maybe will add other later

if __name__ == "__main__":
    f = lambda p: 1 / (np.square(p) + 0.1) + np.square(p) + p[1] / 10
    result = gr.gradient_descent(f, partial(gr.forward_derivative, delta=0.05), gr.get_inv_root_step(0.1),
                                 gr.get_stop_f_eps(0.01),
                                 np.array([-1.9, 1.5]))
    if not result.success:
        print("I'm sorry, no solution")
    else:
        print(result.get_res(), len(result.guesses))
        print(result.history)
