# GO1124-Png
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def sigmoidal(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))


if __name__ == '__main__':
    print("=== Função de Pertinência Sigmoidal ===")
    # PARÂMETROS da sigmoidal: a=inclinação (maior a = transição mais abrupta),
    # c=ponto de inflexão (onde μ=0.5). Para outro domínio, ajuste estes dois valores.
    # Use a>0 para curva crescente (ex: 'velho') e a<0 para decrescente (ex: 'jovem').
    # Parâmetros: inclinação a=1, ponto de inflexão c=5
    a, c = 1, 5
    print(f"  Parâmetros: a={a} (inclinação), c={c} (ponto de inflexão)")
    print()
    xs = list(range(-1, 12))
    mus = [sigmoidal(x, a, c) for x in xs]
    for x, mu in zip(xs, mus):
        barra = "█" * int(mu * 20)
        print(f"  x={x:3d}: μ={mu:.2f} |{barra}")

    # VISUALIZAÇÃO — as linhas de referência mostram c (onde μ=0.5, ponto de virada)
    # e μ=0.5 horizontal. Ideal para fenômenos que aumentam indefinidamente sem limite claro.
    # --- Gráfico colorido ---
    fig, ax = plt.subplots(figsize=(9, 4))

    x_cont = np.linspace(-1, 11, 500)
    mu_cont = sigmoidal(x_cont, a, c)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for i in range(len(x_cont) - 1):
        mu_mid = (mu_cont[i] + mu_cont[i + 1]) / 2
        ax.fill_between(
            x_cont[i:i+2], 0, mu_cont[i:i+2],
            color=cmap(norm(mu_mid)), alpha=0.85, linewidth=0
        )

    ax.plot(x_cont, mu_cont, color='#333333', linewidth=2, label='μ(x) sigmoidal')

    ax.scatter(xs, mus, color=[cmap(norm(m)) for m in mus],
               edgecolors='#333333', s=80, zorder=5, label='Pontos discretos')

    for x, mu in zip(xs, mus):
        if 0.05 < mu < 0.95:
            ax.annotate(f'{mu:.2f}', (x, mu), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color='#222222')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline(0.5, color='orange', linewidth=1, linestyle=':', alpha=0.8, label='μ = 0.5')
    ax.axvline(c, color='steelblue', linewidth=1, linestyle=':',
               alpha=0.7, label=f'Inflexão c={c}  (μ=0.5)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Grau de pertinência μ(x)', fontsize=9)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('μ(x)', fontsize=11)
    ax.set_title(f'Função de Pertinência Sigmoidal  (a={a}, c={c})', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(range(-1, 12))

    plt.tight_layout()
    plt.savefig('GO1124-Png.png', dpi=150)
    plt.show()
    print("\n  Gráfico salvo em GO1124-Png.png")
