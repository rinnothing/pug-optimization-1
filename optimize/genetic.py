import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sqlalchemy.orm.sync import populate

import common.tests_function as t_fun
import common

def selection_min(pop, values, selection_count = 2):
    selected = []
    selection_count = min(selection_count, len(pop))
    for _ in range(len(pop)):
        ind = np.random.randint(0, len(pop), selection_count)
        values_rand_pop = values[ind]
        best_in_rand_count = ind[np.argmin(values_rand_pop)]
        selected.append(pop[best_in_rand_count])
    return selected

def crossover_min(parent1, parent2, crossover_rate = 0.7):
    if np.random.rand() < crossover_rate:
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    return parent1

def mutate_min(child, mutate_rate = 0.1, mutate_delta = 1):
    if np.random.rand() < mutate_rate:
        child += np.random.normal(0, mutate_delta, size=child.shape)
    return child

def create_pop_min(pop_size, lim):
    return np.random.uniform(lim[0], lim[1], (pop_size, 2))

def abs_genetic_algorithm(mutate, crossover, selection, create_pop, fun, lim, pop_size, generations, selection_count = 2, crossover_rate = 0.7, mutate_rate = 0.1, mutate_delta = 1):
    if pop_size % 2 != 0:
        pop_size += 1
    population = create_pop(pop_size, lim)
    res = common.StateResult()
    res.function = fun

    best_ind = population[0]
    best_value = fun(best_ind)
    res.add_function_call()

    for gen in range(generations):
        values = np.array([fun(ind) for ind in population])
        res.add_function_call(pop_size)

        now_best_ind = np.argmin(values)
        now_best_value = values[now_best_ind]
        if now_best_value < best_value:
            best_ind = population[now_best_ind]
            best_value = now_best_value
        res.add_guess(population[now_best_ind])
        res.add_history(population)

        selected = selection(population, values, selection_count)

        new_population = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1 = crossover(parent1, parent2, crossover_rate)
            child2 = crossover(parent2, parent1, crossover_rate)
            child1 = mutate(child1, mutate_rate, mutate_delta)
            child2 = mutate(child2, mutate_rate, mutate_delta)
            new_population.extend([child1, child2])

        population = new_population

    res.add_guess(best_ind)
    res.add_history(population)
    res.success = True
    return res

def genetic_algorithm(fun, lim, pop_size, generations, selection_count = 2, crossover_rate = 0.7, mutate_rate = 0.1, mutate_delta = 1):
    return abs_genetic_algorithm(mutate_min, crossover_min, selection_min, create_pop_min,
                                 fun, lim, pop_size, generations, selection_count, crossover_rate, mutate_rate, mutate_delta)

def visualiser_2d_gif(state, lim_x, lim_y, index):
    X, Y = np.meshgrid(np.linspace(lim_x[0], lim_x[1], 200), np.linspace(lim_y[0], lim_y[1], 200))
    Z = state.function([X, Y])

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='f(x, y)')

    scat = ax.scatter([], [], c='red', s=50, alpha=0.7, label='Особи')
    best_point = ax.scatter([], [], c='green', s=50, alpha=1, label='Лучшая особь')

    ax.set_xlim(lim_x)
    ax.set_ylim(lim_y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    def animate(i):
        best_point.set_offsets([state.guesses[i]])
        scat.set_offsets(state.history[i])
        return scat, best_point

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(state.guesses),
        interval=200,
        repeat=True
    )
    plt.show()

if __name__ == "__main__":
    numb_f = 3
    f = t_fun.functions_with_local_min_2d[numb_f].function
    lim = t_fun.functions_with_local_min_2d[numb_f].lim

    res = genetic_algorithm(f, lim, 50, 100,2, 0.7, 0.1, (lim[1] - lim[0]) / 10)
    visualiser_2d_gif(res, lim, lim, 1)

    print(f"\nНайденный минимум: x={res.get_res()[0]:.4f}, y={res.get_res()[1]:.4f}, f={f(res.get_res()):.4f}")
    values = np.array([f(ind) for ind in res.guesses])
    plt.plot(values)
    plt.show()

