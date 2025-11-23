import numpy as np

def make_fitness(evaluator, yield_strength):
    def fitness(dims):
        result = evaluator.evaluate(dims)

        return {
            "volume": float(result["volume"]),
            "stress": float(result["stress"]),
            "yield_strength": float(yield_strength)
        }
    return fitness

