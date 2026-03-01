# GO1916-NãoRequerInstalaçãoUsaApenasBibliotecas
def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1):
    """Adicionar ruído gaussiano"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, sigma)
    return mutated

def uniform_mutation(individual, mutation_rate=0.1, bounds=(-10, 10)):
    """Substituir por valor aleatório no intervalo"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            mutated[i] = np.random.uniform(bounds[0], bounds[1])
    return mutated
