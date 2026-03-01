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


if __name__ == '__main__':
    import numpy as np
    np.random.seed(2)

    print("=== Operadores de Mutação para Permutações ===")

    rota = [0, 1, 2, 3, 4, 5, 6, 7]
    print(f"  Rota original: {rota}")

    # Swap mutation
    mut_swap = swap_mutation(rota.copy())
    print(f"\n  Swap:      {mut_swap}")
    assert sorted(mut_swap) == sorted(rota), "Permutação inválida!"

    # Inversion mutation
    mut_inv = inversion_mutation(rota.copy())
    print(f"  Inversion: {mut_inv}")
    assert sorted(mut_inv) == sorted(rota), "Permutação inválida!"

    # Scramble mutation
    mut_scr = scramble_mutation(rota.copy())
    print(f"  Scramble:  {mut_scr}")
    assert sorted(mut_scr) == sorted(rota), "Permutação inválida!"

    print("\n  ✅ Todas as mutações preservam a validade da permutação")
