import matplotlib.patches

import common

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def wolfe_conditions(fun, grad, step, x, max_count = 20, c1 = 0.8, c2 = 1e-3):
    res = common.StateResult()
    res.function = fun

    for i in range(1, max_count):
        new_x = x + step
        new_y = fun(new_x)
        y = fun(x)
        grad_x = grad(x)
        grad_new = grad(new_x)
        lower = new_y > y + c1 * np.dot(step, grad_x)
        curvature = np.sum(grad_new) >= np.sum(c2 * grad_x)
        res.add_guess(new_x)

        if not lower and curvature:
            res.success = True
            return res

        if lower:
            step = (1 - 0.5 / i) * step
        else:
            step = (1 + 1 / i) * step

    res.success = True
    return res

def visualiser(state: common.StateResult, x, lim_x, lim_y):
    X, Y = np.meshgrid(np.linspace(lim_x[0],lim_x[1], 200), np.linspace(lim_y[0], lim_y[1],200))
    Z = state.function([X, Y])

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    arrow = matplotlib.patches.Arrow(x[0], x[1], state.guesses[0][0] - x[0], state.guesses[0][1] - x[1], width=0.5, color="red")
    ax.add_patch(arrow)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def animate(i):
        arrow.set_data(x[0], x[1], state.guesses[i][0] - x[0], state.guesses[i][1] - x[1])
        return arrow

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(state.guesses),
        repeat=True,
        interval=300
    )
    plt.show()


def visualiser_1(state: common.StateResult, limits, x, c1, c2, y, y_g, freq=50, path: str = None):
    fig, ax = plt.subplots()

    # setting limits
    ax.set_xlim(limits)

    t = np.arange(limits[0], limits[1], (limits[1] - limits[0]) / freq)

    # drawing the base graphic
    ax.plot(t, state.function(t))

    # adding moving objects
    point, = ax.plot(0, 0, 'ro')

    linec1 = ax.axline((x, y), slope=y_g * c1, color='blue')
    linec2 = ax.axline((x, y), slope=y_g * c2, color='green')
    # making animation function
    def animate(i):
        point.set_data([state.guesses[i]], [state.function(state.guesses[i])])
        return point, linec1, linec2

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(state.guesses),
        repeat=True,
        interval=500
    )

    if path is not None:
        writer = animation.PillowWriter(fps=15, bitrate=1800)
        ani.save(path, writer=writer)

    plt.show()

if __name__ == "__main__":
    lim = [-13, 20]
    f = lambda x: x ** 4 / 100 - x ** 3 / 10 - x ** 2 / 2 + 2 * x + 5
    gr = lambda x: x ** 3 / 25 - x ** 2 * 3 / 10 - x + 2

    f2 = lambda x: 1/(x[0]**2+x[1]**2+0.1) +x[0]**2+x[1]**2 + x[1]/10
    def gr2(x):
        df_dx0 = 2 * x[0] * (1 - 1 / (x[0]**2 + x[1]**2 + 0.1)**2)
        df_dx1 = 2 * x[1] * (1 - 1 / (x[0]**2 + x[1]**2 + 0.1)**2) + 1/10
        return np.array([df_dx0, df_dx1])
    x2 = np.array([-1.5, 1.8])
    step2 = np.array([0.1, -0.5])
    c1 = 0.8
    c2 = 1e-3
    lim_x = [-3, 3]
    lim_y = [-3, 3]
    res2 = wolfe_conditions(f2, gr2, step2, x2, 20, c1, c2)
    visualiser(res2, x2, lim_x, lim_y)

    x1 = np.array(-10)
    step1 = np.array(4)
    res1 = wolfe_conditions(f, gr, step1, x1, 20, c1, c2)
    visualiser_1(res1, lim, x1, c1, c2, f(x1), gr(x1))