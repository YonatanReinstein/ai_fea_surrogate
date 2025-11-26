import numpy as np

def make_fitness(evaluator, yield_strength):
    def fitness(dims):
        result = evaluator.evaluate(dims)

        volumes = [float(volumee) for volumee in result["volume"]]
        stresses = [float(stress) for stress in result["stress"]]

        return {
            "volume": volumes,
            "stress": stresses,
            "yield_strength": float(yield_strength)
        }
    return fitness

