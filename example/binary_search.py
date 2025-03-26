import common

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def binary_search(fun, x=None, step=None, stop=None, bounds=None):
    """
    binary search example, doesn't need x and step, but requires bounds
    """
    res = common.StateResult()
    # putting fun
    res.function = fun

    # need to have bounds for binary search
    if bounds is None:
        return res

    # init bounds
    s, f = bounds
    m = (s + f) // 2

    # you can put values you need to explain previous steps in history
    res.add_history(np.array([s, f]))
    res.add_guess(m)

    # doing basic binary search
    while not stop(res):
        m_res = fun(m)

        # doing binary search stuff
        if m_res <= 0:
            s = m
        else:
            f = m

        # middle is the new guess, need to put it in guesses
        m = (s + f) / 2
        res.add_guess(m)
        res.add_history(np.array([s, f]))

    res.add_guess(m)

    res.success = True
    return res


def get_eps_stop_determiner(eps: float):
    """
    eps bounds difference stopper, only works for binary search
    :param eps: value, that determines what bounds difference it should stop
    :return: to stop or not
    """

    def determiner(state: common.OptimizeResult) -> bool:
        return (state.history[-1][1] - state.history[-1][0]) < eps

    return determiner


def binary_visualiser(state: common.StateResult, limits, freq=50, path: str = None):
    fig, ax = plt.subplots()

    # setting limits
    ax.set_xlim(limits)

    t = np.arange(limits[0], limits[1], (limits[1] - limits[0]) / freq)

    # drawing the base graphic
    ax.plot(t, state.function(t))

    # adding moving objects
    point, = ax.plot(0, 0, 'ro')
    lline = ax.axvline(0, color='orange')
    rline = ax.axvline(0, color='orange')

    # making animation function
    def animate(i):
        point.set_data([state.guesses[i]], [state.function(state.guesses[i])])

        left, right = state.history[i]
        lline.set_xdata([left] * 2)
        rline.set_xdata([right] * 2)

        return point, lline, rline

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(state.history),
        repeat=True,
        interval=100
    )

    if path is not None:
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(path, writer=writer)

    plt.show()


# example on usage of created functions
lim = [6.0, 8.0]
result = binary_search(lambda x: (x - 5.0) ** 2.0 - 5.0, stop=get_eps_stop_determiner(0.1), bounds=lim)
if not result.success:
    print("didn't solve")
else:
    binary_visualiser(result, lim, 200)
