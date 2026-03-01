# GO1919-Slides1822SeleECrossover
# Seleção Torneio
def tournament_selection_tsp(population, cities, k=5):
    fitness_values = [fitness_tsp(ind, cities) for ind in population]
    selected = []
    for _ in range(len(population)):
        competitors = np.random.choice(len(population), k, replace=False)
        winner = max(competitors, key=lambda i: fitness_values[i])
        selected.append(population[winner].copy())
    return selected

# Order Crossover
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child[start:end] = parent1[start:end]
    p2_filtered = [g for g in parent2 if g not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = p2_filtered[idx]
            idx += 1
    return child

# Swap Mutation
def swap_mutation_tsp(individual, mutation_rate=0.2):
    mutated = individual.copy()
    if np.random.random() < mutation_rate:
        i, j = np.random.choice(len(mutated), 2, replace=False)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


if __name__ == '__main__':
    import numpy as np
    np.random.seed(5)

    print("=== Seleção por Torneio + OX + Swap para TSP ===")

    # TSP com 6 cidades geradas aleatoriamente
    n_cities = 6
    cities = np.random.rand(n_cities, 2) * 100

    # Função de fitness TSP (distância total — menor é melhor → usamos -distância)
    def fitness_tsp(route, cities):
        dist = 0
        for i in range(len(route)):
            a, b = cities[route[i]], cities[route[(i+1) % len(route)]]
            dist += np.linalg.norm(a - b)
        return -dist  # negativo para maximizar

    # População inicial de 8 rotas aleatórias
    populacao = [list(np.random.permutation(n_cities)) for _ in range(8)]

    melhor_antes = max(populacao, key=lambda r: fitness_tsp(r, cities))
    print(f"  Melhor rota antes (distância): {-fitness_tsp(melhor_antes, cities):.2f}")
    print(f"  Rota: {melhor_antes}")

    # Um passo de evolução
    selecionados = tournament_selection_tsp(populacao, cities, k=3)
    filho = order_crossover(selecionados[0], selecionados[1])
    filho_mut = swap_mutation_tsp(filho, mutation_rate=0.3)

    dist_filho = -fitness_tsp(filho_mut, cities)
    print(f"\n  Filho após OX + swap: {filho_mut}")
    print(f"  Distância do filho: {dist_filho:.2f}")
