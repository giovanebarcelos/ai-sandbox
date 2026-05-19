# GO1123-Png
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def gaussiana(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))


if __name__ == '__main__':
    print("=== Função de Pertinência Gaussiana ===")
    # PARÂMETROS da gaussiana: c=centro (onde μ=1), sigma=largura da curva.
    # Para outro domínio: c desloca o pico; sigma maior = curva mais larga (menos seletiva).
    # Parâmetros: centro c=5, desvio padrão sigma=2
    c, sigma = 5, 2
    print(f"  Parâmetros: c={c}, σ={sigma}")
    print()
    xs = list(range(-1, 12))
    mus = [gaussiana(x, c, sigma) for x in xs]
    for x, mu in zip(xs, mus):
        barra = "█" * int(mu * 20)
        print(f"  x={x:3d}: μ={mu:.2f} |{barra}")

    # VISUALIZAÇÃO — a região ±σ mostra onde a pertinência cai de 1.0 para ~0.61.
    # Vantagem da gaussiana: transição infinitamente suave, sem cantos (vs triangular).
    # --- Gráfico colorido ---
    fig, ax = plt.subplots(figsize=(9, 4))

    x_cont = np.linspace(-1, 11, 500)
    mu_cont = gaussiana(x_cont, c, sigma)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for i in range(len(x_cont) - 1):
        mu_mid = (mu_cont[i] + mu_cont[i + 1]) / 2
        ax.fill_between(
            x_cont[i:i+2], 0, mu_cont[i:i+2],
            color=cmap(norm(mu_mid)), alpha=0.85, linewidth=0
        )

    ax.plot(x_cont, mu_cont, color='#333333', linewidth=2, label='μ(x) gaussiana')

    ax.scatter(xs, mus, color=[cmap(norm(m)) for m in mus],
               edgecolors='#333333', s=80, zorder=5, label='Pontos discretos')

    for x, mu in zip(xs, mus):
        if mu > 0.05:
            ax.annotate(f'{mu:.2f}', (x, mu), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color='#222222')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline(1, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(c, color='steelblue', linewidth=1, linestyle=':', alpha=0.7, label=f'Centro c={c}')
    # Marcação de ±σ
    ax.axvspan(c - sigma, c + sigma, alpha=0.08, color='steelblue', label=f'±σ = [{c-sigma}, {c+sigma}]')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Grau de pertinência μ(x)', fontsize=9)

    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('μ(x)', fontsize=11)
    ax.set_title(f'Função de Pertinência Gaussiana  (c={c}, σ={sigma})', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(range(-1, 12))

    plt.tight_layout()
    plt.savefig('GO1123-Png.png', dpi=150)
    plt.show()
    print("\n  Gráfico salvo em GO1123-Png.png")
