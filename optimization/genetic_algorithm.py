import random
import os
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_func, dims_dict,
                 pop_size=200, generations=200, 
                 crossover_rate=0.85, mutation_rate=0.8, seed=0):

        self.fitness_func = fitness_func
        self.dims_dict = dims_dict

        self.dim_names = list(dims_dict.keys())
        self.dim = len(self.dim_names)

        self.bounds = np.array([
            [dims_dict[name]["min"], dims_dict[name]["max"]]
            for name in self.dim_names
        ])

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate   # per-individual mutation prob
        self.starting_gen = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def vector_to_dict(self, vec):
        return {name: float(vec[i]) for i, name in enumerate(self.dim_names)}

    def dict_to_vector(self, d):
        return np.array([d[name] for name in self.dim_names], dtype=float)

    def _initialize_population(self):
        population = None
        gen = 0
        while True:
            if os.path.exists(f"optimization/artifacts/ga_population_gen_{gen+1}.npy"):
                population = np.load(f"optimization/artifacts/ga_population_gen_{gen+1}.npy")
                gen += 1
                self.starting_gen = gen
            else:
                break
        if population is not None:
            print("[GA] Loading checkpoint from ga_population.npy")
            return population

        low, high = self.bounds[:, 0], self.bounds[:, 1]
        print("[GA] Initializing new population")
        return np.random.uniform(low, high, (self.pop_size, self.dim))

    def _mutate(self, individual, gen, rate=0.16):
        # per-gene mutation probability
        mask = np.random.rand(self.dim) < rate

        # adaptive sigma: shrinks slowly as search progresses
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        sigma_scale = 0.05# * (0.98 ** gen)   # starts 0.05 → slowly to ~0.01
        sigma = sigma_scale * ranges

        noise = np.random.normal(0.0, sigma, size=self.dim)
        mutated = np.where(mask, individual + noise, individual)
        return np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])

    def _crossover(self, p1, p2):
        if random.random() < self.crossover_rate:
            alpha = np.random.uniform(0.3, 0.7)
            return alpha * p1 + (1 - alpha) * p2
        return p1.copy()

    def _diversity_scores(self, population):
        distances = np.linalg.norm(
            population[:, np.newaxis, :] - population[np.newaxis, :, :],
            axis=2
        )
        mean_distances = np.mean(distances, axis=1)

        if mean_distances.max() > mean_distances.min():
            return (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min())
        return np.zeros_like(mean_distances)

    def _select_parents_tournament(self, fitnesses, population, count, k=3):
        fitnesses = np.array(fitnesses)
        parents = []
        for _ in range(count):
            idxs = np.random.choice(len(population), size=k, replace=False)
            best_idx = idxs[np.argmin(fitnesses[idxs])]
            parents.append(population[best_idx])
        return parents

    def run(self):
        population = self._initialize_population()

        best_vec = None
        best_fit = float("inf")

        for gen in range(self.starting_gen, self.starting_gen + self.generations):

            # ---------- Evaluate ----------
            dims_dicts = [self.vector_to_dict(ind) for ind in population]
            res = self.fitness_func(dims_dicts)

            raw_volume = np.array(res["volume"])
            raw_stress = np.array(res["stress"])

            # ---------- SOFT penalty ----------
            raw_stress_banner = raw_stress.copy()
            violation = np.maximum(raw_stress - res["yield_strength"], 0.0)
            penalty = violation * 1
            raw_volume_penalized = raw_volume + penalty

            # ---------- Diversity ----------
            raw_diversity = self._diversity_scores(population)
            mean_div = raw_diversity.mean()

            # ---------- Normalize ----------
            def norm(x):
                return (x - x.min()) / (x.ptp() + 1e-8)

            V_norm = norm(raw_volume_penalized)
            S_norm = norm(penalty)
            D_norm = norm(raw_diversity)

            #w_v = 0.55
            #w_s = 0.3
            #w_d = 0.15   

            w_v = 0.49
            w_s = 0.49
            w_d = 0.02

            fitnesses = w_v * V_norm + w_s * S_norm + w_d * (1 - D_norm)

            # ---------- Banner: TOP-K BEST INDIVIDUALS ----------
            from .screenshot import screenshot

            top_k = min(1, len(population))  # how many screenshots per gen
            best_indices = np.argsort(fitnesses)[:top_k]

            for rank, idx in enumerate(best_indices):
                banner = (
                    f"[gen {gen:03d} | rank {rank:02d}] "
                    f"rvol: {raw_volume[idx]:.4e}, "
                    f"rstress: {raw_stress_banner[idx]:.4e} "
                    f"=> fit: {fitnesses[idx]:.4e}"
                )
                screenshot(
                    geometry="arm",
                    dims=self.vector_to_dict(population[idx]),
                    save_path=f"optimization/screenshots/gen_{gen:03d}_rank_{rank:02d}_idx_{idx}.png",
                    banner=banner
                )

            # ---------- Global best ----------
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_vec = population[gen_best_idx].copy()

            print(f"Gen {gen+1}/{self.generations} | Best fitness: {fitnesses.min():.4e} | Div={mean_div:.4f}")

            # ---------- Rank ----------
            ranked = sorted(zip(fitnesses, population), key=lambda x: x[0])
            sorted_fit, sorted_pop = zip(*ranked)
            sorted_fit = np.array(sorted_fit)
            sorted_pop = np.array(sorted_pop)

            half = int(self.pop_size * 0.8)
            survivors = sorted_pop[:half]

            # ---------- Tournament selection ----------
            parents_pool = self._select_parents_tournament(
                sorted_fit[:half], survivors, half, k=2
            )

            # ---------- Children ----------
            children = []
            while len(children) < (self.pop_size - half):
                p1, p2 = random.sample(list(parents_pool), 2)
                child = self._crossover(p1, p2)
                children.append(child)

            population = np.vstack([survivors, children])

            # ---------- Mutation (1 elite) ----------
            mutated_population = []
            for i, ind in enumerate(population):
                if i < 4:
                    mutated_population.append(ind)
                else:
                    if random.random() < self.mutation_rate:
                        ind = self._mutate(ind, gen)
                    mutated_population.append(ind)
            population = np.array(mutated_population)

            # ---------- Adaptive mutation rate ----------
            if gen > 100 and gen % 20 == 0:
                old_rate = self.mutation_rate
                self.mutation_rate = min(1.0, self.mutation_rate * 1.15)
                print(f"🔧 Mutation rate boosted: {old_rate:.3f} → {self.mutation_rate:.3f}")

            # ---------- Checkpoint ----------
            np.save(f"optimization/artifacts/ga_population_gen_{gen+1}.npy", population)
        return self.vector_to_dict(best_vec)
