# GO1102-ImplementacaoEmPython
# Operadores fuzzy: AND (T-normas), OR (S-normas), NOT (complemento)
import matplotlib.pyplot as plt
import numpy as np


# ── T-normas (AND) ────────────────────────────────────────────────────────────
def and_min(a, b):
    return min(a, b)

def and_prod(a, b):
    return a * b

def and_bounded(a, b):
    return max(0, a + b - 1)


# ── S-normas (OR) ─────────────────────────────────────────────────────────────
def or_max(a, b):
    return max(a, b)

def or_soma(a, b):
    return a + b - a * b

def or_bounded(a, b):
    return min(1, a + b)


# ── Complemento (NOT) ─────────────────────────────────────────────────────────
def fuzzy_not(a):
    return 1 - a


# ── Relatório formatado ───────────────────────────────────────────────────────
def relatorio(mu_a, mu_b, label_a="A", label_b="B"):
    nao_a = fuzzy_not(mu_a)
    nao_b = fuzzy_not(mu_b)

    print("=" * 52)
    print("  OPERADORES FUZZY — EXEMPLO COMPLETO")
    print("=" * 52)
    print(f"  μ_{label_a}  = {mu_a:.2f}")
    print(f"  μ_{label_b}  = {mu_b:.2f}")
    print()

    print("── AND (T-normas) ──────────────────────────────")
    r_and_min  = and_min(mu_a, mu_b)
    r_and_prod = and_prod(mu_a, mu_b)
    r_and_bnd  = and_bounded(mu_a, mu_b)
    print(f"  Mínimo     : min({mu_a}, {mu_b})           = {r_and_min:.4f}")
    print(f"  Produto    : {mu_a} × {mu_b}               = {r_and_prod:.4f}")
    print(f"  Bounded    : max(0, {mu_a}+{mu_b}-1)       = {r_and_bnd:.4f}")
    print()

    print("── OR (S-normas) ───────────────────────────────")
    r_or_max  = or_max(mu_a, mu_b)
    r_or_soma = or_soma(mu_a, mu_b)
    r_or_bnd  = or_bounded(mu_a, mu_b)
    print(f"  Máximo     : max({mu_a}, {mu_b})           = {r_or_max:.4f}")
    print(f"  Soma Alg.  : {mu_a}+{mu_b}-{mu_a}×{mu_b}  = {r_or_soma:.4f}")
    print(f"  Bounded    : min(1, {mu_a}+{mu_b})         = {r_or_bnd:.4f}")
    print()

    print("── NOT (Complemento) ───────────────────────────")
    print(f"  ¬{label_a}         : 1 - {mu_a}                = {nao_a:.4f}")
    print(f"  ¬{label_b}         : 1 - {mu_b}                = {nao_b:.4f}")
    print()

    print("── REGRA COMPLETA ──────────────────────────────")
    print(f"  SE {label_a} E {label_b}  → AND/min  = {r_and_min:.2f}  ({r_and_min*100:.0f}%)")
    print(f"  SE {label_a} OU {label_b} → OR/max   = {r_or_max:.2f}  ({r_or_max*100:.0f}%)")
    print(f"  SE NÃO {label_a}    → NOT       = {nao_a:.2f}  ({nao_a*100:.0f}%)")
    print("=" * 52)

    return {
        "and_min": r_and_min, "and_prod": r_and_prod, "and_bnd": r_and_bnd,
        "or_max":  r_or_max,  "or_soma":  r_or_soma,  "or_bnd":  r_or_bnd,
        "not_a":   nao_a,     "not_b":    nao_b,
    }


# ── Gráfico comparativo ───────────────────────────────────────────────────────
def grafico(mu_a, mu_b, resultados, label_a="A", label_b="B"):
    labels = [
        "AND\nMín", "AND\nProd", "AND\nBnd",
        "OR\nMáx",  "OR\nSoma", "OR\nBnd",
        f"NOT\n{label_a}", f"NOT\n{label_b}",
    ]
    valores = [
        resultados["and_min"],  resultados["and_prod"], resultados["and_bnd"],
        resultados["or_max"],   resultados["or_soma"],  resultados["or_bnd"],
        resultados["not_a"],    resultados["not_b"],
    ]
    cores = (
        ["#2196F3", "#1976D2", "#90CAF9"] +   # AND — tons de azul
        ["#4CAF50", "#388E3C", "#A5D6A7"] +   # OR  — tons de verde
        ["#FF5722", "#FF8A65"]                # NOT — tons de laranja
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, valores, color=cores, edgecolor="#333", linewidth=0.8, width=0.6)

    # Linha de referência dos inputs
    ax.axhline(mu_a, color="steelblue", linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"μ_{label_a} = {mu_a}")
    ax.axhline(mu_b, color="seagreen",  linewidth=1.2, linestyle="--",
               alpha=0.7, label=f"μ_{label_b} = {mu_b}")

    # Valor acima de cada barra
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Separadores visuais entre grupos
    ax.axvline(2.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axvline(5.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    # Rótulos de grupo
    for xpos, txt, cor in [(1, "T-normas (AND)", "#1565C0"),
                            (4, "S-normas (OR)",  "#2E7D32"),
                            (6.5, "NOT",          "#BF360C")]:
        ax.text(xpos, 1.08, txt, ha="center", fontsize=9,
                color=cor, fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Grau de pertinência μ", fontsize=10)
    ax.set_title(
        f"Comparação de Operadores Fuzzy  "
        f"(μ_{label_a}={mu_a}, μ_{label_b}={mu_b})",
        fontsize=12
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig("GO1102-ImplementacaoEmPython.png", dpi=150)
    plt.show()
    print("\n  Gráfico salvo em GO1102-ImplementacaoEmPython.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # CENÁRIO — μ_QUENTE(28°C)=0.8 e μ_ALTA(75%)=0.6 do slide 7.
    # Para outro problema, substitua QUENTE e ALTA pelos graus das suas variáveis.
    # Cenário: temperatura "quente" e umidade "alta"
    QUENTE = 0.8   # μ_QUENTE(28°C)
    ALTA   = 0.6   # μ_ALTA(75%)

    # Os operadores AND, OR e NOT são genéricos — só os valores de entrada mudam.
    resultados = relatorio(QUENTE, ALTA, label_a="QUENTE", label_b="ALTA")
    grafico(QUENTE, ALTA, resultados, label_a="QUENTE", label_b="ALTA")
