# GO1106-ImplementacaoEmPython
# Métodos de Defuzzificação — converte um conjunto fuzzy de saída em um valor numérico preciso.
# Contexto: após a inferência fuzzy (etapa de Mamdani), o resultado é uma função de pertinência
# agregada (mu). A defuzzificação extrai um único número dessa distribuição.

import numpy as np


def defuzzify_centroid(x, mu):
    """Centroide (Center of Gravity)
    Retorna o 'centro de massa' da área sob a curva mu.
    É o método mais preciso e mais usado na prática.
    Fórmula: x* = Σ(x · μ(x)) / Σμ(x)
    """
    # Numerador: soma ponderada — cada ponto x pesado pela sua pertinência
    # Denominador: área total sob a curva (fator de normalização)
    return np.sum(x * mu) / np.sum(mu)


def defuzzify_bisector(x, mu):
    """Bisector (Bissetriz)
    Retorna o ponto que divide a área sob mu em duas metades iguais.
    Similar ao centroide, mas mais robusto a distribuições assimétricas.
    """
    total_area = np.sum(mu)        # área total da distribuição
    half_area = total_area / 2     # metade da área que queremos encontrar

    cumsum = 0
    for i, m in enumerate(mu):
        cumsum += m                # acumula área da esquerda para a direita
        if cumsum >= half_area:    # encontrou o ponto de bisseção
            return x[i]

    return x[-1]  # fallback: retorna o último ponto se não encontrar antes


def defuzzify_mom(x, mu):
    """Mean of Maximum (MOM) — Média dos Máximos
    Localiza todos os pontos com pertinência máxima e retorna a média entre eles.
    Útil quando o pico é plano (região de máximo com vários pontos).
    """
    max_mu = np.max(mu)                        # valor máximo de pertinência
    max_indices = np.where(mu == max_mu)[0]    # índices onde mu atinge o máximo
    return np.mean(x[max_indices])             # média dos x correspondentes


def defuzzify_som(x, mu):
    """Smallest of Maximum (SOM) — Menor dos Máximos
    Dentre todos os pontos com pertinência máxima, retorna o menor x.
    Produz uma saída conservadora (tendência ao menor valor).
    """
    max_mu = np.max(mu)
    max_indices = np.where(mu == max_mu)[0]
    return x[max_indices[0]]   # primeiro índice = menor x com mu máximo


def defuzzify_lom(x, mu):
    """Largest of Maximum (LOM) — Maior dos Máximos
    Dentre todos os pontos com pertinência máxima, retorna o maior x.
    Produz uma saída agressiva (tendência ao maior valor).
    """
    max_mu = np.max(mu)
    max_indices = np.where(mu == max_mu)[0]
    return x[max_indices[-1]]  # último índice = maior x com mu máximo


# ---------------------------------------------------------------------------
# Exemplo de uso — sistema de gorjeta fuzzy
# Universo de discurso: gorjeta entre 0% e 25%
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Domínio de saída: 100 pontos igualmente espaçados entre 0% e 25%
    x = np.linspace(0, 25, 100)

    # Função de pertinência triangular representando "gorjeta boa"
    # Pico em 15% (pertinência = 1.0), zeros em 5% e 25%
    # Fórmula: max(0, 1 - |x - 15| / 10)
    mu = np.maximum(0, 1 - np.abs(x - 15) / 10)

    # Aplica cada método de defuzzificação sobre a mesma distribuição mu
    gorjeta_centroid  = defuzzify_centroid(x, mu)
    gorjeta_bisector  = defuzzify_bisector(x, mu)
    gorjeta_mom       = defuzzify_mom(x, mu)
    gorjeta_som       = defuzzify_som(x, mu)
    gorjeta_lom       = defuzzify_lom(x, mu)

    print("=== Resultados da Defuzzificação ===")
    print(f"Centroide (COG): {gorjeta_centroid:.1f}%  ← mais preciso, mais usado")
    print(f"Bisector:        {gorjeta_bisector:.1f}%  ← divide a área em 2 partes iguais")
    print(f"MOM:             {gorjeta_mom:.1f}%  ← média dos pontos de máximo")
    print(f"SOM:             {gorjeta_som:.1f}%  ← menor ponto de máximo")
    print(f"LOM:             {gorjeta_lom:.1f}%  ← maior ponto de máximo")
