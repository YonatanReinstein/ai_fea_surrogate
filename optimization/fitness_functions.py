import numpy as np

STRESS_LIMIT = 1e8  # adjust as needed

def make_fitness(evaluator):
    """
    Returns a fitness function that the GA can call.
    """
    def fitness(dims):
        result = evaluator.evaluate(dims)
        stress, disp, volume = result["stress"], result["disp"], result["volume"]
        penalty = 0.0
        if stress > STRESS_LIMIT:
            penalty += (stress / STRESS_LIMIT - 1) ** 2
        return volume * (1 + penalty)
    return fitness
