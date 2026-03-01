# GO1106-ImplementacaoEmPython
import numpy as np

def defuzzify_centroid(x, mu):
    """Centroide"""
    return np.sum(x * mu) / np.sum(mu)

def defuzzify_bisector(x, mu):
    """Bisector"""
    total_area = np.sum(mu)
    half_area = total_area / 2
    cumsum = 0
    for i, m in enumerate(mu):
        cumsum += m
        if cumsum >= half_area:
            return x[i]
    return x[-1]

def defuzzify_mom(x, mu):
    """Mean of Maximum"""
    max_mu = np.max(mu)
    max_indices = np.where(mu == max_mu)[0]
    return np.mean(x[max_indices])

def defuzzify_som(x, mu):
    """Smallest of Maximum"""
    max_mu = np.max(mu)
    max_indices = np.where(mu == max_mu)[0]
    return x[max_indices[0]]

def defuzzify_lom(x, mu):
    """Largest of Maximum"""
    max_mu = np.max(mu)
    max_indices = np.where(mu == max_mu)[0]
    return x[max_indices[-1]]

# Exemplo de uso


if __name__ == "__main__":
    x = np.linspace(0, 25, 100)
    mu = np.array([...])  # valores de pertinencia
    gorjeta_centroid = defuzzify_centroid(x, mu)
    gorjeta_mom = defuzzify_mom(x, mu)
    print(f"Centroide: {gorjeta_centroid:.1f}%")
    print(f"MOM: {gorjeta_mom:.1f}%")
