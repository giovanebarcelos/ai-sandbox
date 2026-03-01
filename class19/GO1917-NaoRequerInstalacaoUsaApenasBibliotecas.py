# GO1917-NãoRequerInstalaçãoUsaApenasBibliotecas
def swap_mutation(individual):
    """Trocar 2 posições"""
    mutated = individual.copy()
    i, j = np.random.choice(len(mutated), 2, replace=False)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated

def inversion_mutation(individual):
    """Inverter segmento"""
    mutated = individual.copy()
    i, j = sorted(np.random.choice(len(mutated), 2, replace=False))
    mutated[i:j] = reversed(mutated[i:j])
    return mutated

def scramble_mutation(individual):
    """Embaralhar segmento"""
    mutated = individual.copy()
    i, j = sorted(np.random.choice(len(mutated), 2, replace=False))
    segment = mutated[i:j]
    np.random.shuffle(segment)
    mutated[i:j] = segment
    return mutated
