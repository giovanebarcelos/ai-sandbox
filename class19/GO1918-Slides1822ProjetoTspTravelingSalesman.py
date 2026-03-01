# GO1918-Slides1822ProjetoTspTravelingSalesman
import numpy as np
import matplotlib.pyplot as plt

# Gerar cidades aleatórias
np.random.seed(42)
N_CITIES = 20
cities = np.random.rand(N_CITIES, 2) * 100  # Coordenadas (x, y) em [0, 100]

# Calcular distância entre duas cidades
def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Calcular distância total da rota
def calculate_route_distance(route, cities):
    total_distance = 0
    for i in range(len(route)):
        city_a = cities[route[i]]
        city_b = cities[route[(i + 1) % len(route)]]  # Volta para início
        total_distance += distance(city_a, city_b)
    return total_distance

# Fitness: maximizar (-distância) para usar max
def fitness_tsp(route, cities):
    return -calculate_route_distance(route, cities)

# População inicial: permutações aleatórias
def create_population_tsp(pop_size, n_cities):
    return [np.random.permutation(n_cities).tolist() for _ in range(pop_size)]
