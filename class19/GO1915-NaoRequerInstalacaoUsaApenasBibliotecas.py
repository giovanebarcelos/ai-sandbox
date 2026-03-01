# GO1915-NãoRequerInstalaçãoUsaApenasBibliotecas
def order_crossover(parent1, parent2):
    """OX: preserva ordem relativa"""
    size = len(parent1)
    child = [-1] * size

    # Escolher subseção do parent1
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child[start:end] = parent1[start:end]

    # Preencher resto com ordem do parent2
    p2_filtered = [g for g in parent2 if g not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = p2_filtered[idx]
            idx += 1
    return child

def partially_mapped_crossover(parent1, parent2):
    """PMX: troca segmentos e mapeia conflitos"""
    size = len(parent1)
    child = parent1.copy()

    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    mapping = {}

    # Trocar segmento
    for i in range(start, end):
        mapping[parent2[i]] = parent1[i]
        child[i] = parent2[i]

    # Resolver conflitos fora do segmento
    for i in range(size):
        if i < start or i >= end:
            while child[i] in child[start:end]:
                child[i] = mapping[child[i]]

    return child


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    print("=== Operadores de Crossover para Problemas de Permutação ===")

    # Exemplo: rota de TSP com 8 cidades (0–7)
    parent1 = [0, 1, 2, 3, 4, 5, 6, 7]
    parent2 = [3, 5, 7, 0, 2, 6, 4, 1]

    print(f"  parent1: {parent1}")
    print(f"  parent2: {parent2}")

    # OX (Order Crossover)
    filho_ox = order_crossover(parent1, parent2)
    print(f"\n  OX filho:  {filho_ox}")
    assert len(set(filho_ox)) == 8, "OX: permutação inválida!"
    print("    ✅ Permutação válida (sem repetição)")

    # PMX (Partially Mapped Crossover)
    filho_pmx = partially_mapped_crossover(parent1, parent2)
    print(f"\n  PMX filho: {filho_pmx}")
    assert len(set(filho_pmx)) == 8, "PMX: permutação inválida!"
    print("    ✅ Permutação válida (sem repetição)")
