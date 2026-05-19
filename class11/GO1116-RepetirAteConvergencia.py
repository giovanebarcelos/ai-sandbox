# GO1116-RepetirAteConvergencia
# Algoritmo Genético (GA) para otimização automática dos parâmetros das MFs.
# Problema: definir os parâmetros (a, b, c) de cada MF manualmente é difícil
# quando não há especialista disponível. O GA busca a combinação que minimiza
# o erro em dados de teste.
#
# Como funciona: cada 'individual' é uma lista de parâmetros de MFs
# [a1,b1,c1, a2,b2,c2, ...]. O GA evolui a população minimizando o MSE.
#
# Para usar em um problema real, implemente:
#   - create_fuzzy(individual): monta o sistema fuzzy com os parâmetros
#   - test_fuzzy(sys, X_test, y_test): calcula o erro do sistema nos dados
#   - population: população inicial gerada com toolbox.population()
# Requer: pip install deap
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
