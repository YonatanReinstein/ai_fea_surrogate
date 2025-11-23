import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_func, dims_dict,
                 pop_size=30, generations=40, 
                 crossover_rate=0.5, mutation_rate=0.4, seed=None):
        self.checkpoint_file = "ga_population.npy"
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
        import os

        # if checkpoint exists → load population instead of initializing
        if os.path.exists(self.checkpoint_file):
            print(f"[GA] Loading checkpoint from {self.checkpoint_file}")
            return np.load(self.checkpoint_file)

        # otherwise random init
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        print("[GA] Initializing new population")
        return np.random.uniform(low, high, (self.pop_size, self.dim))

    def _mutate(self, individual, rate=0.3):
        mask = np.random.rand(self.dim) < rate

        # Random mutation factors only for selected genes
        factors = np.random.uniform(0.8, 1.2, size=self.dim)

        # Apply mutation where mask == True
        mutated = np.where(mask, individual * factors, individual)

        # Enforce parameter bounds
        return np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])

    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            alpha = np.random.uniform(0.3, 0.7)
            return alpha * p1 + (1 - alpha) * p2
        return p1.copy()
    
    def _diversity_scores(self, population):
        # Compute pairwise distances
        distances = np.linalg.norm(
            population[:, np.newaxis, :] - population[np.newaxis, :, :],
            axis=2
        )

        # Distance of each ind. to all others (exclude self-distance = 0)
        mean_distances = np.mean(distances, axis=1)

        # Normalize between 0 and 1
        if mean_distances.max() > mean_distances.min():
            normalized = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min())
        else:
            normalized = np.zeros_like(mean_distances)

        return normalized


    def run(self):
        population = self._initialize_population()
        with open("ga_log.txt", "w") as log_file:
            best = None
            for gen in range(self.generations):
                raw_volume = []
                raw_stress = []
                for ind in population:
                    res = self.fitness_func(self.vector_to_dict(ind))
                    raw_volume.append(res["volume"])
                    raw_stress.append(res["stress"])

                raw_volume = np.array(raw_volume)
                raw_stress = np.array(raw_stress)
                raw_stress = raw_stress + np.maximum(100 * (raw_stress - res["yield_strength"]), 0)
    

                # Compute diversity for current generation
                raw_diversity = self._diversity_scores(population)

                V_min = raw_volume.min()
                V_max = raw_volume.max()
                V_norm = (raw_volume - V_min) / (V_max - V_min + 1e-8)


                S_min = raw_stress.min()
                S_max = raw_stress.max()
                S_norm = (raw_stress - S_min) / (S_max - S_min + 1e-8)

                D_min = raw_diversity.min()
                D_max = raw_diversity.max()
                D_norm = (raw_diversity - D_min) / (D_max - D_min + 1e-8)

                w_v = 0.35
                w_s = 0.55
                w_d = 0.1

                fitnesses = w_v * V_norm + w_s * S_norm + w_d * D_norm

                log_file.write(f"generation{gen+1    }, best fitness: {fitnesses.min()}\n")  
                for i, (ind, fit) in enumerate(zip(population, fitnesses)):
                    log_file.write(f", fitness: {fit/fitnesses.sum()} stress: {S_norm[i]/S_norm.sum()} volume: {V_norm[i]/V_norm.sum()}\n")
                    log_file.flush()
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
                # Save checkpoint every 50 generations
                if (gen + 1) % 10 == 0:
                    np.save(self.checkpoint_file, population)
                    log_file.write(f"[Checkpoint] Saved population at generation {gen+1}\n")
                    log_file.flush()
                    print(f"[GA] Saved checkpoint at generation {gen+1}")


        # return vector AND dict form
        best_fit, best_vec = best
        return self.vector_to_dict(best_vec)
