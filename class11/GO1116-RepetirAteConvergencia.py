# GO1116-RepetirAteConvergencia
from deap import base, creator, tools, algorithms
import random

# Definir fitness (minimizar erro)


if __name__ == "__main__":
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Função de avaliação
    def evaluate_fuzzy_system(individual):
        # individual = [a1,b1,c1, a2,b2,c2, ...]
        fuzzy_sys = create_fuzzy(individual)  # Criar sistema com parâmetros
        mse = test_fuzzy(fuzzy_sys, X_test, y_test)  # Testar em dados
        return (mse,)

    # Executar GA
    pop, log = algorithms.eaSimple(population, toolbox,
                                    cxpb=0.7, mutpb=0.2,
                                    ngen=50)
