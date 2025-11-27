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
        # per-gene mutation mask
        mask = np.random.rand(self.dim) < rate

        # scale of mutation ~ 10% of range for each gene
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        sigma = 0.15 * ranges

        noise = np.random.normal(0.0, sigma, size=self.dim)

        mutated = np.where(mask, individual + noise, individual)
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

        best_vec = None
        best_fit = float("inf")

        for gen in range(self.generations):

            # ----- evaluate individuals -----
            dims_dicts = [self.vector_to_dict(ind) for ind in population]
            res = self.fitness_func(dims_dicts)

            raw_volume = np.array(res["volume"])
            raw_stress = np.array(res["stress"])

            # penalize yield violation
            raw_stress = np.maximum(100 * (raw_stress - res["yield_strength"]), 0)

            # diversity
            raw_diversity = self._diversity_scores(population)

            # ----- normalize -----
            V_norm = (raw_volume - raw_volume.min()) / (raw_volume.ptp() + 1e-8)
            S_norm = (raw_stress - raw_stress.min()) / (raw_stress.ptp() + 1e-8)
            D_norm = (raw_diversity - raw_diversity.min()) / (raw_diversity.ptp() + 1e-8)

            # weights
            w_v = 0.4
            w_s = 0.4
            w_d = 0.2

            # fitness (lower = better)
            fitnesses = w_v * V_norm + w_s * S_norm + w_d * D_norm

            # ----- update global best -----
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_vec = population[gen_best_idx].copy()

            # log
            print(f"Gen {gen+1}/{self.generations} | Best fitness: {fitnesses.min():.4e}")

            # ----- rank individuals -----
            ranked = sorted(zip(fitnesses, population), key=lambda x: x[0])
            _, sorted_pop = zip(*ranked)
            sorted_pop = np.array(sorted_pop)

            half = self.pop_size // 2

            # ---- TOP HALF survive but may mutate ----
            survivors = []
            for indiv in sorted_pop[:half]:
                if random.random() < self.mutation_rate:
                    survivors.append(self._mutate(indiv))
                else:
                    survivors.append(indiv)
            survivors = np.array(survivors)

            # ---- Parents for crossover come only from TOP HALF ----
            parents_pool = survivors

            # ---- BOTTOM HALF replaced by children ----
            children = []
            while len(children) < (self.pop_size - half):
                p1, p2 = random.sample(list(parents_pool), 2)
                child = self._crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                children.append(child)

            # ---- New generation ----
            population = np.vstack([survivors, children])

            # checkpoint
            if (gen + 1) % 2 == 0:
                np.save(self.checkpoint_file, population)

        return self.vector_to_dict(best_vec)

    def _select_parents_proportional(self, fitnesses, population, count):
        fitnesses = np.array(fitnesses)

        # convert to scores where high = good
        max_f = fitnesses.max()
        scores = max_f - fitnesses + 1e-8

        prob = scores / scores.sum()

        idxs = np.random.choice(len(population), size=count, p=prob, replace=True)
        return [population[i] for i in idxs]
    
    def _select_parents_tournament(self, fitnesses, population, count, k=3):
        fitnesses = np.array(fitnesses)
        parents = []
        for _ in range(count):
            # randomly sample k candidates
            idxs = np.random.choice(len(population), size=k, replace=False)
            # pick the one with the lowest fitness (remember: lower is better)
            best_idx = idxs[np.argmin(fitnesses[idxs])]
            parents.append(population[best_idx])
        return parents


