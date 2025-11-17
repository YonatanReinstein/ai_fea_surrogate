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


def make_fitness(evaluator, 
                 stress_limit=None, 
                 disp_limit=None, 
                 volume_weight=0.5, 
                 stress_weight=0.25,
                 disp_weight=0.25,
                 penalty_factor=10.0):



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

        fitness_value = (
            volume_weight * volume +
            stress_weight * stress +
            disp_weight   * disp
        ) * (1 + penalty)
        return fitness_value

    return fitness

