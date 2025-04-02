import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def save_final_graph(state, limits, index, freq=50, y_limits=None):
    fig, ax = plt.subplots()

    ax.set_xlim(limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    t = np.linspace(limits[0], limits[1], freq)
    ax.plot(t, state.function(t), label="Функция")

    guesses = np.array(state.guesses)
    ax.plot(guesses, state.function(guesses), 'r-', linewidth=2, label="Траектория")
    ax.scatter(guesses, state.function(guesses), color='red', s=30, label="Точки пути")
    ax.plot(guesses[-1], state.function(guesses[-1]), 'bo', markersize=8, label="Финальная точка")

    ax.legend()

    save_path = f"result_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def save_final_grad(state, lim_x, lim_y, index):
    fig, ax = plt.subplots(figsize=(10, 8))

    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z = state.function([X, Y])

    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    guesses = np.array(state.guesses)

    ax.plot(guesses[:, 0], guesses[:, 1], 'r-', linewidth=2, label="Траектория")
    ax.scatter(guesses[:, 0], guesses[:, 1], color='red', s=30, label="Точки пути")
    ax.plot(guesses[-1, 0], guesses[-1, 1], 'bo', markersize=8, label="Финальная точка")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    save_path = f"grad_{index}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
