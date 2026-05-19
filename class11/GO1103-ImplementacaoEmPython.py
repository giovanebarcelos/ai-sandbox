# GO1103-ImplementacaoEmPython
# Variáveis linguísticas com funções de pertinência trapezoidais
#
# Parâmetros de TEMPERATURA escolhidos para T=18°C:
#   FRIO:      max(0, min(1, (22-x)/10))  → μ(18) = (22-18)/10 = 0.4
#   AGRADÁVEL: max(0, min((x-12)/10, 1, (35-x)/10)) → μ(18) = (18-12)/10 = 0.6
#   QUENTE:    max(0, min(1, (x-25)/10))  → μ(18) = 0.0
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class VariavelLinguistica:
    def __init__(self, nome, universo):
        self.nome = nome
        self.universo = universo
        self.termos = {}

    def adicionar_termo(self, nome, funcao):
        self.termos[nome] = funcao

    def avaliar(self, valor):
        return {t: f(valor) for t, f in self.termos.items()}


def relatorio(var, valor):
    resultado = var.avaliar(valor)
    print(f"=== Variável Linguística: {var.nome} ===")
    print(f"Universo: {var.universo} | T = {valor}°C\n")
    for termo, mu in resultado.items():
        barra = "█" * int(mu * 20)
        print(f"  μ_{termo:<10}({valor}) = {mu:.2f}  |{barra}")
    print()
    total = sum(resultado.values())
    print(f"  Soma dos graus: {total:.2f}")
    return resultado


def grafico(var, t_exemplo, resultado, saida):
    x = np.linspace(*var.universo, 500)
    cores  = {"FRIO": "#3A7BD5", "AGRADÁVEL": "#2ECC71", "QUENTE": "#E74C3C"}
    alphas = {"FRIO": 0.25,      "AGRADÁVEL": 0.20,      "QUENTE": 0.20}

    fig, ax = plt.subplots(figsize=(10, 5))

    for termo, func in var.termos.items():
        y = np.array([func(xi) for xi in x])
        cor = cores[termo]
        ax.plot(x, y, color=cor, linewidth=2.5, label=termo)
        ax.fill_between(x, 0, y, color=cor, alpha=alphas[termo])

    # Linha vertical em T=t_exemplo
    ax.axvline(t_exemplo, color="purple", linewidth=1.5,
               linestyle="--", label=f"T = {t_exemplo}°C")

    # Pontos e anotações em T=t_exemplo
    for termo, mu in resultado.items():
        if mu > 0:
            cor = cores[termo]
            ax.scatter(t_exemplo, mu, color=cor, edgecolors="#222", s=80, zorder=5)
            ax.annotate(
                f"μ_{termo} = {mu:.1f}",
                (t_exemplo, mu),
                xytext=(t_exemplo + 1.2, mu),
                fontsize=9, color=cor,
                arrowprops=dict(arrowstyle="-", color=cor, lw=0.8),
            )

    ax.set_xlim(var.universo)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Temperatura (°C)", fontsize=11)
    ax.set_ylabel("Grau de Pertinência (μ)", fontsize=11)
    ax.set_title(f"Variável Linguística: {var.nome}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.set_xticks(range(0, 55, 5))

    plt.tight_layout()
    plt.savefig(saida, dpi=150)
    plt.show()
    print(f"  Gráfico salvo em {saida}")


if __name__ == "__main__":
    temp = VariavelLinguistica("TEMPERATURA", (0, 50))
    temp.adicionar_termo("FRIO",
        lambda x: max(0.0, min(1.0, (22 - x) / 10)))
    temp.adicionar_termo("AGRADÁVEL",
        lambda x: max(0.0, min((x - 12) / 10, 1.0, (35 - x) / 10)))
    temp.adicionar_termo("QUENTE",
        lambda x: max(0.0, min(1.0, (x - 25) / 10)))

    T = 18
    resultado = relatorio(temp, T)

    # Salva em images/ dois níveis acima do script; se não existir, salva no cwd
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    raiz = os.path.abspath(os.path.join(base, "..", "..", "images"))
    if not os.path.isdir(raiz):
        raiz = os.getcwd()
    saida = os.path.join(raiz, "Aula1146.png")
    grafico(temp, T, resultado, saida)
