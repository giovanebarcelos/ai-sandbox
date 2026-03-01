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


if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    print("=== Operadores de Mutação para Cromossomos Reais ===")

    individuo = np.array([1.5, 3.0, -2.0, 0.5, 4.0])
    print(f"  Individual original: {individuo}")

    # Mutação Gaussiana
    mut_gauss = gaussian_mutation(individuo.copy(), mutation_rate=0.5, sigma=0.3)
    print(f"\n  Após mutação gaussiana (rate=0.5, sigma=0.3):")
    print(f"    {mut_gauss.round(4)}")

    # Mutação Uniforme
    mut_unif = uniform_mutation(individuo.copy(), mutation_rate=0.5, bounds=(-5.0, 5.0))
    print(f"\n  Após mutação uniforme (rate=0.5, bounds=(-5,5)):")
    print(f"    {mut_unif.round(4)}")
