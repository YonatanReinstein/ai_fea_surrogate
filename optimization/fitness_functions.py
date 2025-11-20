import numpy as np

def make_fitness(evaluator, stress_limit, disp_limit=None, penalty_factor=100.0):
    def fitness(dims):
        result = evaluator.evaluate(dims)
        stress = float(result["stress"])
        disp   = float(result.get("disp", 0.0))
        volume = float(result["volume"])
        penalty = 0.0
        if stress_limit is not None and stress > stress_limit:
            penalty += penalty_factor * (stress / stress_limit - 1) ** 2
        if disp_limit is not None and disp > disp_limit:
            penalty += penalty_factor * (disp / disp_limit - 1) ** 2
        fitness = volume * (1 + penalty)
        return fitness
    return fitness

