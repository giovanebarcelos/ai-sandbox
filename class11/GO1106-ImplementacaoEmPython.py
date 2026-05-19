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
    # Função de pertinência triangular com pico em 15% (gorjeta "boa")
    mu = np.maximum(0, 1 - np.abs(x - 15) / 10)
    gorjeta_centroid = defuzzify_centroid(x, mu)
    gorjeta_mom = defuzzify_mom(x, mu)
    gorjeta_bisector = defuzzify_bisector(x, mu)
    gorjeta_som = defuzzify_som(x, mu)
    gorjeta_lom = defuzzify_lom(x, mu)
    print(f"Centroide: {gorjeta_centroid:.1f}%")
    print(f"MOM:       {gorjeta_mom:.1f}%")
    print(f"Bisector:  {gorjeta_bisector:.1f}%")
    print(f"SOM:       {gorjeta_som:.1f}%")
    print(f"LOM:       {gorjeta_lom:.1f}%")
