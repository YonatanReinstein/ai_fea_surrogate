import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_func, dims_dict, 
                 pop_size=30, generations=40, 
                 crossover_rate=0.5, mutation_rate=0.2, seed=None):

        self.fitness_func = fitness_func
        self.dims_dict = dims_dict

        # dimension ordering
        self.dim_names = list(dims_dict.keys())
        self.dim = len(self.dim_names)

        # bounds array (dim × 2)
        self.bounds = np.array([
            [dims_dict[name]["min"], dims_dict[name]["max"]]
            for name in self.dim_names
        ])

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def vector_to_dict(self, vec):
        """Convert vector → {dim_name: value}"""
        return {name: float(vec[i]) for i, name in enumerate(self.dim_names)}

    def dict_to_vector(self, d):
        """Convert {name: value} → vector"""
        return np.array([d[name] for name in self.dim_names], dtype=float)

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
            # evaluate fitness on dict form
            fitnesses = np.array([
                self.fitness_func(self.vector_to_dict(ind))
                for ind in population
            ])

            ranked = sorted(zip(fitnesses, population), key=lambda x: x[0])
            best_fit, best_vec = ranked[0]
            best = (best_fit, best_vec)

            print(f"Gen {gen+1}/{self.generations} | Best fitness: {best_fit:.4e}")

            # elitism: keep top 50%
            survivors = [x[1] for x in ranked[:self.pop_size // 2]]

            # offspring generation
            offspring = []
            while len(offspring) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = self._crossover(p1, p2)

                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                offspring.append(child)

            population = np.array(offspring)

        # return vector AND dict form
        best_fit, best_vec = best
        return self.vector_to_dict(best_vec)
