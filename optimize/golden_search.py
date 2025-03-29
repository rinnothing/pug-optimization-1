import common

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def golden_search(fun, max_count_step=1000000, stop=None, bounds=None):
    res = common.StateResult()
    # putting fun
    res.function = fun

    # need to have bounds for binary search
    if bounds is None:
        return res

    # init bounds
    s, f = bounds
    # init Golden ratio
    gold = (np.sqrt(5) + 1) / 2

    # create left and right points
    l = f - (f - s) / gold
    r = s + (f - s) / gold
    fun_l = fun(l)
    fun_r = fun(r)

    # you can put values you need to explain previous steps in history
    res.add_history(np.array([s, f]))

    # doing golden search
    while not stop(res) and max_count_step:
        max_count_step -= 1
        if fun_l < fun_r:
            f = r
            my_choose = r
            r = l
            fun_r = fun_l
            l = f - (f - s) / gold
            fun_l = fun(l)
        else:
            s = l
            my_choose = l
            l = r
            fun_l = fun_r
            r = s + (f - s) / gold
            fun_r = fun(r)

        res.add_guess(my_choose)
        res.add_history(np.array([s, f]))

    res.add_guess((s + f) / 2)

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


def visualiser(state: common.StateResult, limits, freq=50, path: str = None):
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
        interval=500
    )

    if path is not None:
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(path, writer=writer)

    plt.show()


# example on usage of created functions
if __name__ == "__main__":
    lim = [-5, 8.0]
    result = golden_search(lambda x: (x + 2) ** 2.0 - 5.0, stop=get_eps_stop_determiner(0.1), bounds=lim)
    if not result.success:
        print("didn't solve")
    visualiser(result, lim, 200)
