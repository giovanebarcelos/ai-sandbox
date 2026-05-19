# GO1101-Png
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


if __name__ == '__main__':
    print("=== Função de Pertinência Triangular ===")
    # PARÂMETROS — pé esquerdo, pico e pé direito da triangular. Para outro
    # domínio, ajuste: a=início do suporte, b=valor com grau 1, c=fim do suporte.
    # Parâmetros: pé esquerdo a=0, pico b=5, pé direito c=10
    a, b, c = 0, 5, 10
    print(f"  Parâmetros: a={a}, b={b}, c={c}")
    print()
    xs = list(range(-1, 12))
    mus = [triangular(x, a, b, c) for x in xs]
    for x, mu in zip(xs, mus):
        barra = "█" * int(mu * 20)
        print(f"  x={x:3d}: μ={mu:.2f} |{barra}")

    # VISUALIZAÇÃO — gradiente RdYlGn mostra visualmente quão 'pertencente'
    # cada ponto é ao conjunto. Para outro conjunto, apenas os params acima mudam.
    # --- Gráfico colorido ---
    fig, ax = plt.subplots(figsize=(9, 4))

    # Curva contínua preenchida com gradiente de cor por grau de pertinência
    x_cont = np.linspace(-1, 11, 500)
    mu_cont = np.array([triangular(xi, a, b, c) for xi in x_cont])

    cmap = plt.cm.RdYlGn          # vermelho (baixo) → amarelo → verde (alto)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Preenche fatias verticais coloridas de acordo com μ
    for i in range(len(x_cont) - 1):
        mu_mid = (mu_cont[i] + mu_cont[i + 1]) / 2
        ax.fill_between(
            x_cont[i:i+2], 0, mu_cont[i:i+2],
            color=cmap(norm(mu_mid)), alpha=0.85, linewidth=0
        )

    # Linha da curva
    ax.plot(x_cont, mu_cont, color='#333333', linewidth=2, label='μ(x) triangular')

    # Pontos discretos usados na representação em caractere
    ax.scatter(xs, mus, color=[cmap(norm(m)) for m in mus],
               edgecolors='#333333', s=80, zorder=5, label='Pontos discretos')

    # Anotações dos graus nos pontos
    for x, mu in zip(xs, mus):
        if mu > 0:
            ax.annotate(f'{mu:.1f}', (x, mu), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color='#222222')

    # Linhas de referência
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(b, color='steelblue', linewidth=1, linestyle=':', alpha=0.7, label=f'Pico b={b}')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Grau de pertinência μ(x)', fontsize=9)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('μ(x)', fontsize=11)
    ax.set_title(f'Função de Pertinência Triangular  (a={a}, b={b}, c={c})', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(range(-1, 12))

    plt.tight_layout()
    plt.savefig('GO1101-Png.png', dpi=150)
    plt.show()
    print("\n  Gráfico salvo em GO1101-Png.png")
