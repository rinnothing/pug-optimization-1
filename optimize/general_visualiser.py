import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import saving_png as savepng
import matplotlib

matplotlib.use('TkAgg')

def visualiser(state, limits, index, freq=50, interval=100, l=1, y_limits=None, path=None, display=True):
    fig, ax = plt.subplots()

    ax.set_xlim(limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    t = np.linspace(limits[0], limits[1], freq)
    ax.plot(t, state.function(t))

    trajectory, = ax.plot([], [], 'r-', linewidth=2)

    path_points = ax.scatter([], [], color='red', s=30)

    point, = ax.plot(0, 0, 'bo', markersize=8)

    arrow = None
    if len(state.history) != 0:
        if isinstance(state.history[0], np.ndarray) and len(state.history[0]) == 2:
            lline = ax.axvline(0, color='orange')
            rline = ax.axvline(0, color='orange')

    def animate(i):
        nonlocal arrow
        if arrow:
            arrow.remove()

        guesses = np.array(state.guesses)

        trajectory.set_data(guesses[:i+1], state.function(guesses[:i+1]))
        path_points.set_offsets(np.c_[guesses[:i+1], state.function(guesses[:i+1])])
        point.set_data([guesses[i]], [state.function(guesses[i])])

        if len(state.history) != 0:
            if isinstance(state.history[i], np.ndarray) and len(state.history[i]) == 2:
                left, right = state.history[i]
                lline.set_xdata([left] * 2)
                rline.set_xdata([right] * 2)
                return trajectory, path_points, point, lline, rline
            elif len(state.history[i]) == 2:
                grad_val, step = state.history[i]
                arrow = ax.arrow(
                    guesses[i], state.function(guesses[i]),
                    -grad_val * step * l, -grad_val * grad_val * step * l,
                    head_width=0.5, head_length=0.5, fc='blue', ec='black'
                )
                return trajectory, path_points, point, arrow
        return trajectory, path_points, point

    ani = animation.FuncAnimation(
        fig, animate, frames=len(state.guesses), repeat=True, interval=interval
    )

    if path is not None:
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(path, writer=writer)

    savepng.save_final_graph(state, limits, index)

    if display:
        plt.show()