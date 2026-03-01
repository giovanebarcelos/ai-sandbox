# GO1914-NãoRequerInstalaçãoUsaApenasBibliotecas
def blend_crossover(parent1, parent2, alpha=0.5):
    """BLX-α: interpolar entre pais"""
    child1 = []
    child2 = []
    for g1, g2 in zip(parent1, parent2):
        min_g = min(g1, g2)
        max_g = max(g1, g2)
        range_g = max_g - min_g

        # Expandir intervalo com alpha
        lower = min_g - alpha * range_g
        upper = max_g + alpha * range_g

        child1.append(np.random.uniform(lower, upper))
        child2.append(np.random.uniform(lower, upper))
    return child1, child2

def simulated_binary_crossover(parent1, parent2, eta=20):
    """SBX: similar a crossover binário mas para reais"""
    # eta controla dispersão (alto = filhos próximos dos pais)
    # Usado em algoritmos evolutivos modernos
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1)):
        if np.random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-6:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
    return child1, child2
