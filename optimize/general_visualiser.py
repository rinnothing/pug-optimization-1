import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib

matplotlib.use('TkAgg')


def visualiser(state, limits, freq=50, interval=100, l=1, y_limits=None, path=None, display=True):
    fig, ax = plt.subplots()

    # Установка пределов
    ax.set_xlim(limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    t = np.linspace(limits[0], limits[1], freq)
    ax.plot(t, state.function(t))

    point, = ax.plot(0, 0, 'ro')
    arrow = None
    if isinstance(state.history[0], np.ndarray) and len(state.history[0]) == 2:
        lline = ax.axvline(0, color='orange')
        rline = ax.axvline(0, color='orange')

    def animate(i):
        nonlocal arrow
        if arrow:
            arrow.remove()

        guess = state.guesses[i]
        point.set_data([guess], [state.function(guess)])

        if isinstance(state.history[i], np.ndarray) and len(state.history[i]) == 2:
            left, right = state.history[i]
            lline.set_xdata([left] * 2)
            rline.set_xdata([right] * 2)
            return point, lline, rline
        elif len(state.history[i]) == 2:
            grad_val, step = state.history[i]
            arrow = ax.arrow(guess, state.function(guess), -grad_val * step * l, -grad_val * grad_val * step * l,
                             head_width=0.5, head_length=0.5, fc='blue', ec='black')
            return point, arrow

    ani = animation.FuncAnimation(
        fig, animate, frames=len(state.history), repeat=True, interval=interval
    )

    if path is not None:
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(path, writer=writer)

    if display:
        plt.show()
