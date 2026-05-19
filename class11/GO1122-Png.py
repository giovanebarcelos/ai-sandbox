# GO1122-Png
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def trapezoidal(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif x <= b:
        return (x - a) / (b - a)
    elif x <= c:
        return 1
    else:
        return (d - x) / (d - c)


if __name__ == '__main__':
    print("=== Função de Pertinência Trapezoidal ===")
    # Parâmetros: pé esquerdo a=0, subida b=3, descida c=7, pé direito d=10
    a, b, c, d = 0, 3, 7, 10
    print(f"  Parâmetros: a={a}, b={b}, c={c}, d={d}")
    print()
    xs = list(range(-1, 12))
    mus = [trapezoidal(x, a, b, c, d) for x in xs]
    for x, mu in zip(xs, mus):
        barra = "█" * int(mu * 20)
        print(f"  x={x:3d}: μ={mu:.2f} |{barra}")

    # --- Gráfico colorido ---
    fig, ax = plt.subplots(figsize=(9, 4))

    x_cont = np.linspace(-1, 11, 500)
    mu_cont = np.array([trapezoidal(xi, a, b, c, d) for xi in x_cont])

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for i in range(len(x_cont) - 1):
        mu_mid = (mu_cont[i] + mu_cont[i + 1]) / 2
        ax.fill_between(
            x_cont[i:i+2], 0, mu_cont[i:i+2],
            color=cmap(norm(mu_mid)), alpha=0.85, linewidth=0
        )

    ax.plot(x_cont, mu_cont, color='#333333', linewidth=2, label='μ(x) trapezoidal')

    ax.scatter(xs, mus, color=[cmap(norm(m)) for m in mus],
               edgecolors='#333333', s=80, zorder=5, label='Pontos discretos')

    for x, mu in zip(xs, mus):
        if mu > 0:
            ax.annotate(f'{mu:.1f}', (x, mu), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color='#222222')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.8, linestyle='--')
    ax.axvspan(b, c, alpha=0.12, color='steelblue', label=f'Núcleo [{b}, {c}] (μ=1)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Grau de pertinência μ(x)', fontsize=9)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('μ(x)', fontsize=11)
    ax.set_title(f'Função de Pertinência Trapezoidal  (a={a}, b={b}, c={c}, d={d})', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(range(-1, 12))

    plt.tight_layout()
    plt.savefig('GO1122-Png.png', dpi=150)
    plt.show()
    print("\n  Gráfico salvo em GO1122-Png.png")
