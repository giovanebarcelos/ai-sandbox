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
