import numpy as np

def make_fitness(evaluator, yield_strength):
    def fitness(dims):
        result = evaluator.evaluate(dims)

        # Evaluators (e.g. MAPDL) may return None for samples that failed all
        # retries — keep them as None so the optimizer can penalize them.
        volumes = [None if v is None else float(v) for v in result["volume"]]
        stresses = [None if s is None else float(s) for s in result["stress"]]

        return {
            "volume": volumes,
            "stress": stresses,
            "yield_strength": float(yield_strength)
        }
    return fitness

