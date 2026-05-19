# GO1108-SistemaDeControle
# Reescrito sem skfuzzy — usa apenas numpy e matplotlib.
# Implementa Mamdani completo: fuzzificação → inferência → agregação → defuzzificação.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


def trimf(x, abc):
    """Função de pertinência triangular.
    Sobe linearmente de a até b (pico=1), desce de b até c.
    Casos especiais: a==b (rampa descendente) e b==c (rampa ascendente).
    """
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    if b > a:
        mask = (x >= a) & (x <= b)
        y[mask] = (x[mask] - a) / (b - a)
    if c > b:
        mask = (x >= b) & (x <= c)
        y[mask] = (c - x[mask]) / (c - b)
    y[x == b] = 1.0  # garante pico exato em b
    return y


def defuzzify_centroid(x, mu):
    """Centroide: retorna o centro de massa da área sob mu."""
    denom = np.sum(mu)
    return np.sum(x * mu) / denom if denom > 0 else float(x[len(x) // 2])


if __name__ == "__main__":
    # BLOCO 1 — UNIVERSOS: definem os intervalos de cada variável.
    # Para outro problema: ajuste os arange() para o seu domínio.
    qual_u = np.arange(0, 11, 1)   # qualidade: 0–10
    serv_u = np.arange(0, 11, 1)   # serviço:   0–10
    gorj_u = np.arange(0, 26, 1)   # gorjeta:   0–25%

    # BLOCO 2 — MFs (equivalentes a automf(3) + redefinição manual de qualidade)
    # Cada variável tem 3 termos triangulares que cobrem o universo uniformemente.
    qual_ruim  = trimf(qual_u, [0, 0, 5])
    qual_media = trimf(qual_u, [0, 5, 10])
    qual_boa   = trimf(qual_u, [5, 10, 10])

    serv_poor    = trimf(serv_u, [0, 0, 5])
    serv_average = trimf(serv_u, [0, 5, 10])
    serv_good    = trimf(serv_u, [5, 10, 10])

    # automf(3) no intervalo [0, 25]: breakpoints em 0, 12.5, 25
    gorj_poor    = trimf(gorj_u, [0,    0,    12.5])
    gorj_average = trimf(gorj_u, [0,    12.5, 25  ])
    gorj_good    = trimf(gorj_u, [12.5, 25,   25  ])

    # Visualizar as três variáveis
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, u, mfs, labels, title in [
        (axes[0], qual_u,
         [qual_ruim, qual_media, qual_boa],
         ['ruim', 'média', 'boa'], 'Qualidade'),
        (axes[1], serv_u,
         [serv_poor, serv_average, serv_good],
         ['poor', 'average', 'good'], 'Serviço'),
        (axes[2], gorj_u,
         [gorj_poor, gorj_average, gorj_good],
         ['poor', 'average', 'good'], 'Gorjeta'),
    ]:
        for mf, label, color in zip(mfs, labels, ['b', 'g', 'r']):
            ax.plot(u, mf, color, label=label)
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
    plt.tight_layout()
    plt.show()

    # BLOCO 3 — ENTRADAS DE TESTE (exemplo do slide)
    # Para outro problema: substitua os nomes e valores abaixo.
    q_val = 6.5
    s_val = 9.8

    # Fuzzificação: interpola o grau de pertinência de cada entrada em cada termo
    q_ruim_    = np.interp(q_val, qual_u, qual_ruim)
    q_media_   = np.interp(q_val, qual_u, qual_media)
    q_boa_     = np.interp(q_val, qual_u, qual_boa)

    s_poor_    = np.interp(s_val, serv_u, serv_poor)
    s_average_ = np.interp(s_val, serv_u, serv_average)
    s_good_    = np.interp(s_val, serv_u, serv_good)

    # BLOCO 4 — INFERÊNCIA MAMDANI
    # OR=max, AND=min; implicação=min (corta o consequente no nível alpha)
    # R1: qualidade RUIM OR serviço POOR → gorjeta POOR
    alpha1 = max(q_ruim_, s_poor_)
    # R2: serviço AVERAGE → gorjeta AVERAGE
    alpha2 = s_average_
    # R3: serviço GOOD OR qualidade BOA → gorjeta GOOD
    alpha3 = max(s_good_, q_boa_)

    # Implicação: corta cada consequente no nível alpha da sua regra
    implied_poor    = np.minimum(alpha1, gorj_poor)
    implied_average = np.minimum(alpha2, gorj_average)
    implied_good    = np.minimum(alpha3, gorj_good)

    # Agregação: união (max) de todas as implicações → saída fuzzy final
    aggregated = np.maximum(np.maximum(implied_poor, implied_average), implied_good)

    # Defuzzificação por centroide → valor numérico da gorjeta
    gorjeta_val = defuzzify_centroid(gorj_u, aggregated)
    print(f"Gorjeta: {gorjeta_val:.1f}%")

    # Visualizar resultado com saída agregada e centroide
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gorj_u, gorj_poor,    'b--', alpha=0.4, label='poor')
    ax.plot(gorj_u, gorj_average, 'g--', alpha=0.4, label='average')
    ax.plot(gorj_u, gorj_good,    'r--', alpha=0.4, label='good')
    ax.fill_between(gorj_u, aggregated, alpha=0.3, color='purple', label='Saída Agregada')
    ax.axvline(gorjeta_val, color='k', linestyle='--', label=f'Centroide: {gorjeta_val:.1f}%')
    ax.set_title('Defuzzificação — Gorjeta')
    ax.set_xlabel('Gorjeta (%)')
    ax.set_ylabel('Pertinência')
    ax.legend()
    plt.tight_layout()
    plt.show()
