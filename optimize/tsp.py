import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib import animation

import optimize.genetic as gen
import optimize.simulated_annealing as sim_an

def generate_points(n, map_size):
    points = np.random.rand(n, 2) * map_size
    return points

def distance_matrix(points):
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.linalg.norm(points[i] - points[j])
    return matrix


def create_route_len(matrix):
    def route_length(route):
        length = 0
        n = len(route)
        for i in range(n):
            length += matrix[route[i]][route[(i + 1) % n]]
        return length
    return route_length

def create_route(n):
    route = list(range(n))
    random.shuffle(route)
    return route

def create_pop(pop_size, n):
    population = []
    for _ in range(pop_size):
        population.append(create_route(n))
    return population

def order_crossover(parent1, parent2, nothing = 0):
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b] = parent1[a:b]
    ind = b
    for city in parent2[b:] + parent2[:b]:
        if city not in child[a:b]:
            child[ind % n] = city
            ind += 1
    return child


def mutate_tsp(route, mutation_rate=0.5, nothing = 2):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route


def gen_tsp(fun, n, pop_size, generations, selection_count = 12, crossover_rate = 0.7, mutate_rate = 0.5):
    return gen.abs_genetic_algorithm(mutate_tsp, order_crossover, gen.selection_min, create_pop,
                                     fun, n, pop_size, generations, selection_count, crossover_rate, mutate_rate)


def animate_tsp(state, points):
    new_guesses = [state.guesses[0]]
    for guess in state.guesses:
        if new_guesses[-1] != guess:
            new_guesses.append(guess)
    fig, ax = plt.subplots(figsize=(10, 6))
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())
    new_guesses.append(state.get_res())

    def animate(i):
        ax.clear()
        route = new_guesses[i]
        route_coords = points[route + [route[0]]]

        ax.scatter(points[:, 0], points[:, 1], c='red', s=100)
        ax.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=1)

        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), fontsize=12, ha='center', va='center')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid()

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(new_guesses),
        interval=50,
        repeat=True
    )
    plt.show()
    return ani

def get_next_route(route):
    count = random.randint(1, len(route) // 4)
    new_route = route.copy()
    for _ in range(count):
        i, j = random.sample(range(len(route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


if __name__ == "__main__":
    ns = [5, 10, 15, 20, 40, 80, 100]
    for n in ns:
        sum_g = 0
        sum_s = 0
        max_g = -1.0
        max_s = -1.0
        for _ in range(100):
            points = generate_points(n, 100)
            dist_matrix = distance_matrix(points)
            f = create_route_len(dist_matrix)

            res = gen_tsp(f, n, 93, 130, 10, 0.85, 0.6)
            #animate_tsp(res, cities)
            res2 = sim_an.simulated_annealing(f, create_route(n), get_next_route, 1624,  0.988, 0.016,  4315)
            #animate_tsp(res2, cities)
            rg = f(res.get_res())
            rs = f(res2.get_res())
            sum_g += rg
            sum_s += rs
            abs_m = rs - rg
            if abs_m >= 0:
                max_g = max(max_g, abs_m)
            if abs_m <= 0:
                max_s = max(max_s, -abs_m)
        print("=====================================")
        print(n)
        print(sum_g, " | ", max_g)
        print(sum_s, " | ", max_s)
