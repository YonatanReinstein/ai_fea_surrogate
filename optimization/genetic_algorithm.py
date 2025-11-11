import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_func, dim, bounds, pop_size=30, generations=40, 
                 crossover_rate=0.5, mutation_rate=0.2, seed=None):
        self.fitness_func = fitness_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _initialize_population(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return np.random.uniform(low, high, (self.pop_size, self.dim))

    def _mutate(self, individual):
        idx = random.randint(0, self.dim - 1)
        factor = np.random.uniform(0.8, 1.2)
        individual[idx] *= factor
        return np.clip(individual, self.bounds[:, 0], self.bounds[:, 1])

    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            alpha = np.random.uniform(0.3, 0.7)
            return alpha * p1 + (1 - alpha) * p2
        return p1.copy()

    def run(self):
        population = self._initialize_population()
        best = None
        for gen in range(self.generations):
            fitnesses = np.array([self.fitness_func(ind) for ind in population])
            ranked = sorted(zip(fitnesses, population), key=lambda x: x[0])
            best_fit, best_ind = ranked[0]
            best = (best_fit, best_ind)
            print(f"Gen {gen+1}/{self.generations} | Best fitness: {best_fit:.4e}")

            # selection (elitism)
            survivors = [x[1] for x in ranked[:self.pop_size // 2]]

            # generate offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self._crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                offspring.append(child)

            population = np.array(offspring)
        return best
