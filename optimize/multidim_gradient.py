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

def visualiser_2d(state, lim_x, lim_y):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z = state.function([X, Y])

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    guesses = np.array(state.guesses)

    line, = ax.plot([], [], 'r-', linewidth=2)
    trail, = ax.plot([], [], 'ro', markersize=3)
    point, = ax.plot([], [], 'ro', markersize=6)

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def animate(i):
        line.set_data(guesses[:i+1, 0], guesses[:i+1, 1])
        trail.set_data(guesses[:i+1, 0], guesses[:i+1, 1])
        point.set_data([guesses[i, 0]], [guesses[i, 1]])
        return line, trail, point

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(guesses),
        interval=200,
        repeat=True
    )

    plt.show()

def visualiser_2d_gif(state, lim_x, lim_y, index):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z = state.function([X, Y])

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    guesses = np.array(state.guesses)

    line, = ax.plot([], [], 'r-', linewidth=2)
    trail, = ax.plot([], [], 'ro', markersize=3)
    point, = ax.plot([], [], 'ro', markersize=6)

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def animate(i):
        line.set_data(guesses[:i+1, 0], guesses[:i+1, 1])
        trail.set_data(guesses[:i+1, 0], guesses[:i+1, 1])
        point.set_data([guesses[i, 0]], [guesses[i, 1]])
        return line, trail, point

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(guesses),
        interval=200,
        repeat=True
    )
    gif_path = f"visual_2d_gif{index}.gif"
    ani.save(gif_path, writer='pillow', fps=5)
    plt.show()

def visualiser_2d_in(state, lim_x, lim_y, index=0):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z = state.function([X, Y])

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    guesses = np.array(state.guesses)

    line, = ax.plot([], [], 'r-', linewidth=2)
    trail, = ax.plot([], [], 'ro', markersize=3)
    point, = ax.plot([], [], 'ro', markersize=6)

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def animate(i):
        line.set_data(guesses[:i+1, 0], guesses[:i+1, 1])
        trail.set_data(guesses[:i+1, 0], guesses[:i+1, 1])
        point.set_data([guesses[i, 0]], [guesses[i, 1]])
        return line, trail, point

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(guesses),
        interval=200,
        repeat=True
    )

    animate(len(guesses) - 1)
    save_path = f"visual_2d_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)

def visualiser_2d_dual(state1, state2, lim_x, lim_y, index):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200),
                       np.linspace(lim_y[0], lim_y[1], 200))

    Z1 = state1.function([X, Y])
    Z2 = state2.function([X, Y])

    contour1 = ax1.contourf(X, Y, Z1, levels=20, cmap='viridis', alpha=0.7)
    contour2 = ax2.contourf(X, Y, Z2, levels=20, cmap='plasma', alpha=0.7)

    fig.colorbar(contour1, ax=ax1, label='f₁(x, y)')
    fig.colorbar(contour2, ax=ax2, label='f₂(x, y)')

    for ax in (ax1, ax2):
        ax.set_xlim(lim_x)
        ax.set_ylim(lim_y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    ax1.set_title("Function 1")
    ax2.set_title("Function 2")

    guesses1 = np.array(state1.guesses)
    guesses2 = np.array(state2.guesses)

    line1, = ax1.plot([], [], 'r-', linewidth=2)
    trail1, = ax1.plot([], [], 'ro', markersize=3)
    point1, = ax1.plot([], [], 'ro', markersize=6)

    line2, = ax2.plot([], [], 'r-', linewidth=2)
    trail2, = ax2.plot([], [], 'ro', markersize=3)
    point2, = ax2.plot([], [], 'ro', markersize=6)

    max_frames = max(len(guesses1), len(guesses2))

    def animate(i):
        i1 = min(i, len(guesses1) - 1)
        i2 = min(i, len(guesses2) - 1)

        line1.set_data(guesses1[:i1+1, 0], guesses1[:i1+1, 1])
        trail1.set_data(guesses1[:i1+1, 0], guesses1[:i1+1, 1])
        point1.set_data([guesses1[i1, 0]], [guesses1[i1, 1]])

        line2.set_data(guesses2[:i2+1, 0], guesses2[:i2+1, 1])
        trail2.set_data(guesses2[:i2+1, 0], guesses2[:i2+1, 1])
        point2.set_data([guesses2[i2, 0]], [guesses2[i2, 1]])

        return line1, trail1, point1, line2, trail2, point2

    ani = animation.FuncAnimation(
        fig, animate,
        frames=max_frames,
        interval=200,
        repeat=False
    )

    plt.tight_layout()
    animate(len(guesses1) - 1)
    animate(len(guesses2) - 1)
    save_path = f"visual_2d_dual_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)

def visualiser_2d_2_on1(state1, state2, lim_x, lim_y, index):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z1 = state1.function([X, Y])
    Z2 = state2.function([X, Y])

    fig, ax = plt.subplots(figsize=(8, 6))

    contour1 = ax.contourf(X, Y, Z1, levels=20, cmap='viridis', alpha=0.7, zorder=0)
    contour2 = ax.contourf(X, Y, Z2, levels=20, cmap='plasma', alpha=0.3, zorder=1)

    plt.colorbar(contour1, ax=ax)


    guesses1 = np.array(state1.guesses)
    guesses2 = np.array(state2.guesses)

    line1, = ax.plot([], [], 'r-', linewidth=2, label="Траектория 1", zorder=3)
    trail1, = ax.plot([], [], 'ro', markersize=3, zorder=3)
    point1, = ax.plot([], [], 'ro', markersize=6, zorder=3)

    line2, = ax.plot([], [], 'b-', linewidth=2, label="Траектория 2", zorder=2)
    trail2, = ax.plot([], [], 'bo', markersize=3, zorder=2)
    point2, = ax.plot([], [], 'bo', markersize=6, zorder=2)

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    total_frames = max(len(guesses1), len(guesses2))

    def animate(i):
        if i < len(guesses1):
            line1.set_data(guesses1[:i + 1, 0], guesses1[:i + 1, 1])
            trail1.set_data(guesses1[:i + 1, 0], guesses1[:i + 1, 1])
            point1.set_data([guesses1[i, 0]], [guesses1[i, 1]])
        if i < len(guesses2):
            line2.set_data(guesses2[:i + 1, 0], guesses2[:i + 1, 1])
            trail2.set_data(guesses2[:i + 1, 0], guesses2[:i + 1, 1])
            point2.set_data([guesses2[i, 0]], [guesses2[i, 1]])

        return line1, trail1, point1, line2, trail2, point2

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=200,
        repeat=True
    )

    if len(guesses1) > 0:
        line1.set_data(guesses1[:, 0], guesses1[:, 1])
        trail1.set_data(guesses1[:, 0], guesses1[:, 1])
        point1.set_data([guesses1[-1, 0]], [guesses1[-1, 1]])

    if len(guesses2) > 0:
        line2.set_data(guesses2[:, 0], guesses2[:, 1])
        trail2.set_data(guesses2[:, 0], guesses2[:, 1])
        point2.set_data([guesses2[-1, 0]], [guesses2[-1, 1]])

    save_path = f"visual_2d_2on1_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)


# maybe will add other later
def visualiser_3d(state: common.StateResult, lim_x, lim_y):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 100), np.linspace(lim_y[0], lim_y[1], 100))
    Z = state.function([X, Y])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    guesses = np.array(state.guesses)
    ax.plot(guesses[:, 0], guesses[:, 1], state.function(guesses.T), 'ro-', markersize=5)

    point, = ax.plot([], [], [], 'bo', markersize=8)

    def animate(i):
        point.set_data_3d([guesses[i, 0]], [guesses[i, 1]], [state.function(guesses[i])])
        return point,

    ani = animation.FuncAnimation(fig, animate, frames=len(state.guesses), interval=100, blit=True, repeat=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    plt.show()

def visualiser_3d_in(state: common.StateResult, lim_x, lim_y, index):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 100), np.linspace(lim_y[0], lim_y[1], 100))
    Z = state.function([X, Y])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    guesses = np.array(state.guesses)
    ax.plot(guesses[:, 0], guesses[:, 1], state.function(guesses.T), 'ro-', markersize=5)

    point, = ax.plot([], [], [], 'bo', markersize=8)

    def animate(i):
        point.set_data_3d([guesses[i, 0]], [guesses[i, 1]], [state.function(guesses[i])])
        return point,

    ani = animation.FuncAnimation(fig, animate, frames=len(state.guesses), interval=100, blit=True, repeat=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    animate(len(guesses) - 1)
    save_path = f"visual_3d_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)

def visualiser_3d_dual(state1, state2, lim_x, lim_y, index):
    fig = plt.figure(figsize=(8, 4))

    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 100), np.linspace(lim_y[0], lim_y[1], 100))
    Z1 = state1.function([X, Y])
    Z2 = state2.function([X, Y])

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.6)
    guesses1 = np.array(state1.guesses)
    ax1.plot(guesses1[:, 0], guesses1[:, 1], state1.function(guesses1.T), 'ro-', markersize=5)

    ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.6)
    guesses2 = np.array(state2.guesses)
    ax2.plot(guesses2[:, 0], guesses2[:, 1], state2.function(guesses2.T), 'ro-', markersize=5)

    point1, = ax1.plot([], [], [], 'bo', markersize=8)
    point2, = ax2.plot([], [], [], 'bo', markersize=8)

    def animate1(i):
        point1.set_data_3d([guesses1[i, 0]], [guesses1[i, 1]], [state1.function(guesses1[i])])
        return point1,

    def animate2(i):
        point2.set_data_3d([guesses2[i, 0]], [guesses2[i, 1]], [state2.function(guesses2[i])])
        return point2,

    ani1 = animation.FuncAnimation(fig, animate1, frames=len(state1.guesses), interval=100, blit=True, repeat=False)
    ani2 = animation.FuncAnimation(fig, animate2, frames=len(state2.guesses), interval=100, blit=True, repeat=False)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('f(X, Y)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X, Y)')

    animate1(len(state1.guesses) - 1)
    animate2(len(state2.guesses) - 1)

    save_path = f"visual_3d_dual_{index}.png"
    plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def visualiser_3d_2_on_1(state1, state2, lim_x, lim_y, index):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 100), np.linspace(lim_y[0], lim_y[1], 100))

    Z1 = state1.function([X, Y])
    Z2 = state2.function([X, Y])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.6)

    guesses1 = np.array(state1.guesses)
    guesses2 = np.array(state2.guesses)
    ax.plot(guesses1[:, 0], guesses1[:, 1], state1.function(guesses1.T), 'ro-', markersize=5, label='Траектория 1',
            zorder=2)

    ax.plot(guesses2[:, 0], guesses2[:, 1], state2.function(guesses2.T), 'bo-', markersize=5, label='Траектория 2',
            zorder=1)

    point1, = ax.plot([], [], [], 'ro', markersize=8, zorder=2)
    point2, = ax.plot([], [], [], 'bo', markersize=8, zorder=1)


    def animate(i):
        if i < len(guesses1):
            point1.set_data(guesses1[i:i + 1, 0], guesses1[i:i + 1, 1])
            point1.set_3d_properties(state1.function(guesses1[i:i + 1].T))
        if i < len(guesses2):
            point2.set_data(guesses2[i:i + 1, 0], guesses2[i:i + 1, 1])
            point2.set_3d_properties(state2.function(guesses2[i:i + 1].T))
        return point1, point2

    ani = animation.FuncAnimation(fig, animate, frames=max(len(guesses1), len(guesses2)), interval=100, blit=True,
                                  repeat=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')

    if len(guesses1) > 0:
        x1_last, y1_last = guesses1[-1]
        z1_last = state1.function(guesses1[-1])
        point1.set_data([x1_last], [y1_last])
        point1.set_3d_properties([z1_last])

    if len(guesses2) > 0:
        x2_last, y2_last = guesses2[-1]
        z2_last = state2.function(guesses2[-1])
        point2.set_data([x2_last], [y2_last])
        point2.set_3d_properties([z2_last])

    ax.legend()

    save_path = f"visual_3d_2_on_1_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)

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
            # visualiser_2d(result, lim_x=lim, lim_y=lim)
            # visualiser_3d(result, lim_x=lim, lim_y=lim)
