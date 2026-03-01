# GO1906-6PopulaçãoInicial
import numpy as np

def create_population_binary(pop_size, chromosome_length):
    """População binária aleatória"""
    return [np.random.randint(0, 2, chromosome_length).tolist() 
            for _ in range(pop_size)]

def create_population_real(pop_size, n_genes, bounds):
    """População real aleatória dentro de limites"""
    # bounds = [(min, max), (min, max), ...]
    population = []
    for _ in range(pop_size):
        individual = [np.random.uniform(low, high) 
                      for low, high in bounds]
        population.append(individual)
    return population

def create_population_permutation(pop_size, n_cities):
    """População de permutações para TSP"""
    return [np.random.permutation(n_cities).tolist() 
            for _ in range(pop_size)]

# Exemplo: otimizar função com 5 parâmetros reais em [-10, 10]
pop_size = 100
n_genes = 5
bounds = [(-10, 10)] * n_genes
population = create_population_real(pop_size, n_genes, bounds)

# population[0] = [-3.2, 7.5, -1.1, 4.8, -6.3]  (um indivíduo)
