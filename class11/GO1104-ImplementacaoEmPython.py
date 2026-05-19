# GO1104-ImplementacaoEmPython
# Regras Fuzzy SE-ENTÃO — base de 9 regras (Temperatura × Umidade → AC)
#
# Funções validadas para T=27°C, U=60%:
#   μ_TEMP_MÉDIA(27)  = min(1.7, 0.3) = 0.3
#   μ_TEMP_QUENTE(27) = 0.7
#   μ_UMID_MÉDIA(60)  = min(1.4, 0.6) = 0.6
#   μ_UMID_ALTA(60)   = 0.4
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch


# BLOCO 1 — FUNÇÕES DE PERTINÊNCIA: definem como cada valor numérico
# mapeia para termos linguísticos. Para outro problema, substitua estas 6
# funções pelos termos do seu domínio (ex: velocidade, pressão, pH…).
# ── Funções de pertinência ────────────────────────────────────────────────────
# Temperatura [0, 50]°C
def temp_fria(x):   return max(0.0, min(1.0, (20 - x) / 10))
def temp_media(x):  return max(0.0, min((x - 10) / 10, (30 - x) / 10))
def temp_quente(x): return max(0.0, min(1.0, (x - 20) / 10))

# Umidade [0, 100]%
def umid_baixa(x):  return max(0.0, min(1.0, (50 - x) / 25))
def umid_media(x):  return max(0.0, min((x - 25) / 25, (75 - x) / 25))
def umid_alta(x):   return max(0.0, min(1.0, (x - 50) / 25))


# ── Classe RegraFuzzy ─────────────────────────────────────────────────────────
class RegraFuzzy:
    def __init__(self, numero, cond_temp, cond_umid, consequente):
        self.numero = numero
        self.cond_temp = cond_temp
        self.cond_umid = cond_umid
        self.consequente = consequente

    def avaliar(self, mu_temp, mu_umid):
        return min(mu_temp, mu_umid)

    def __repr__(self):
        return (f"R{self.numero}: SE temp={self.cond_temp} E "
                f"umid={self.cond_umid} → AC {self.consequente}")


# BLOCO 2 — BASE DE REGRAS: todas as 9 combinações (3 temp × 3 umid).
# Para outro problema: redefina as regras de acordo com o conhecimento
# especialista do seu domínio. O total de regras = nº_termos_A × nº_termos_B.
# ── Base de regras ────────────────────────────────────────────────────────────
BASE_REGRAS = [
    RegraFuzzy(1, "FRIA",   "BAIXA", "DESLIGADO"),
    RegraFuzzy(2, "FRIA",   "MÉDIA", "DESLIGADO"),
    RegraFuzzy(3, "FRIA",   "ALTA",  "DESLIGADO"),
    RegraFuzzy(4, "MÉDIA",  "BAIXA", "FRACO"),
    RegraFuzzy(5, "MÉDIA",  "MÉDIA", "MÉDIO"),
    RegraFuzzy(6, "MÉDIA",  "ALTA",  "FORTE"),
    RegraFuzzy(7, "QUENTE", "BAIXA", "FORTE"),
    RegraFuzzy(8, "QUENTE", "MÉDIA", "FORTE"),
    RegraFuzzy(9, "QUENTE", "ALTA",  "MÁXIMO"),
]

FUNCS_TEMP = {"FRIA": temp_fria, "MÉDIA": temp_media, "QUENTE": temp_quente}
FUNCS_UMID = {"BAIXA": umid_baixa, "MÉDIA": umid_media, "ALTA": umid_alta}


# ── Sistema ───────────────────────────────────────────────────────────────────
def fuzzificar(T, U):
    return (
        {t: f(T) for t, f in FUNCS_TEMP.items()},
        {t: f(U) for t, f in FUNCS_UMID.items()},
    )


def avaliar_regras(T, U):
    graus_t, graus_u = fuzzificar(T, U)
    ativacoes = []
    for r in BASE_REGRAS:
        mu_t = graus_t[r.cond_temp]
        mu_u = graus_u[r.cond_umid]
        forca = r.avaliar(mu_t, mu_u)
        ativacoes.append((r, mu_t, mu_u, forca))
    return graus_t, graus_u, ativacoes


def relatorio(T, U):
    graus_t, graus_u, ativacoes = avaliar_regras(T, U)

    print("=" * 60)
    print("  SISTEMA FUZZY — BASE DE 9 REGRAS (Temp × Umid → AC)")
    print("=" * 60)
    print(f"  Entradas: Temperatura = {T}°C  |  Umidade = {U}%\n")

    print("── Fuzzificação ──────────────────────────────────────────")
    for termo, mu in graus_t.items():
        print(f"  μ_TEMP_{termo:<7}({T}) = {mu:.2f}  {'█'*int(mu*20)}")
    print()
    for termo, mu in graus_u.items():
        print(f"  μ_UMID_{termo:<7}({U}) = {mu:.2f}  {'█'*int(mu*20)}")
    print()

    print("── Ativação das Regras ───────────────────────────────────")
    for r, mu_t, mu_u, forca in ativacoes:
        status = f"α={forca:.2f}  {'█'*int(forca*20)}" if forca > 0 else "INATIVA"
        print(f"  R{r.numero}: min({mu_t:.1f},{mu_u:.1f})={forca:.2f}  "
              f"→ AC {r.consequente:<10}  {status}")
    print("=" * 60)

    return graus_t, graus_u, ativacoes


# ── Gráfico ───────────────────────────────────────────────────────────────────
COR_CONS = {
    "DESLIGADO": "#95A5A6",
    "FRACO":     "#3498DB",
    "MÉDIO":     "#2ECC71",
    "FORTE":     "#F39C12",
    "MÁXIMO":    "#E74C3C",
}

TERMOS_T = ["FRIA", "MÉDIA", "QUENTE"]
TERMOS_U = ["BAIXA", "MÉDIA", "ALTA"]
GRID_CONS = [
    ["DESL.", "DESL.", "DESL."],
    ["FRACO", "MÉDIO", "FORTE"],
    ["FORTE", "FORTE", "MÁX."],
]
GRID_CONS_FULL = [
    ["DESLIGADO", "DESLIGADO", "DESLIGADO"],
    ["FRACO",     "MÉDIO",     "FORTE"],
    ["FORTE",     "FORTE",     "MÁXIMO"],
]


def grafico(T, U, graus_t, graus_u, ativacoes, saida):
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax_t  = fig.add_subplot(gs[0, 0])
    ax_u  = fig.add_subplot(gs[0, 1])
    ax_hm = fig.add_subplot(gs[0, 2])
    ax_b  = fig.add_subplot(gs[1, :])

    # ── Temperatura ──────────────────────────────────────────
    x_t = np.linspace(0, 50, 400)
    cores_t = {"FRIA": "#3A7BD5", "MÉDIA": "#2ECC71", "QUENTE": "#E74C3C"}
    for termo, func in FUNCS_TEMP.items():
        y = [func(v) for v in x_t]
        cor = cores_t[termo]
        ax_t.plot(x_t, y, color=cor, lw=2, label=termo)
        ax_t.fill_between(x_t, y, alpha=0.12, color=cor)
    ax_t.axvline(T, color="purple", lw=1.5, ls="--", label=f"T={T}°C")
    for termo, mu in graus_t.items():
        if mu > 0:
            ax_t.scatter(T, mu, color=cores_t[termo], edgecolors="#222", s=70, zorder=5)
            ax_t.annotate(f"{mu:.1f}", (T, mu), xytext=(T + 1, mu + 0.05),
                          fontsize=8, color=cores_t[termo])
    ax_t.set_xlim(0, 50); ax_t.set_ylim(-0.05, 1.2)
    ax_t.set_xlabel("Temperatura (°C)", fontsize=9)
    ax_t.set_ylabel("μ", fontsize=10)
    ax_t.set_title("Temperatura", fontsize=10, fontweight="bold")
    ax_t.legend(fontsize=8); ax_t.grid(axis="y", alpha=0.3)

    # ── Umidade ───────────────────────────────────────────────
    x_u = np.linspace(0, 100, 400)
    cores_u = {"BAIXA": "#3A7BD5", "MÉDIA": "#2ECC71", "ALTA": "#E74C3C"}
    for termo, func in FUNCS_UMID.items():
        y = [func(v) for v in x_u]
        cor = cores_u[termo]
        ax_u.plot(x_u, y, color=cor, lw=2, label=termo)
        ax_u.fill_between(x_u, y, alpha=0.12, color=cor)
    ax_u.axvline(U, color="purple", lw=1.5, ls="--", label=f"U={U}%")
    for termo, mu in graus_u.items():
        if mu > 0:
            ax_u.scatter(U, mu, color=cores_u[termo], edgecolors="#222", s=70, zorder=5)
            ax_u.annotate(f"{mu:.1f}", (U, mu), xytext=(U + 2, mu + 0.05),
                          fontsize=8, color=cores_u[termo])
    ax_u.set_xlim(0, 100); ax_u.set_ylim(-0.05, 1.2)
    ax_u.set_xlabel("Umidade (%)", fontsize=9)
    ax_u.set_ylabel("μ", fontsize=10)
    ax_u.set_title("Umidade", fontsize=10, fontweight="bold")
    ax_u.legend(fontsize=8); ax_u.grid(axis="y", alpha=0.3)

    # ── Heatmap grade de regras ───────────────────────────────
    grid = np.array([
        [min(graus_t[ct], graus_u[cu]) for cu in TERMOS_U]
        for ct in TERMOS_T
    ])
    im = ax_hm.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax_hm.set_xticks(range(3)); ax_hm.set_xticklabels(TERMOS_U, fontsize=8)
    ax_hm.set_yticks(range(3)); ax_hm.set_yticklabels(TERMOS_T, fontsize=8)
    ax_hm.set_xlabel("Umidade", fontsize=9)
    ax_hm.set_ylabel("Temperatura", fontsize=9)
    ax_hm.set_title("Grade de Regras (α = min)", fontsize=10, fontweight="bold")
    for i in range(3):
        for j in range(3):
            v = grid[i, j]
            txt = f"{GRID_CONS[i][j]}\nα={v:.2f}"
            cor_txt = "white" if v > 0.55 else "#222"
            ax_hm.text(j, i, txt, ha="center", va="center",
                       fontsize=8, color=cor_txt, fontweight="bold")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="α")

    # ── Barras de ativação ────────────────────────────────────
    rotulos = [f"R{r.numero}\n{r.cond_temp[:3]}/\n{r.cond_umid[:3]}"
               for r, *_ in ativacoes]
    forcas  = [forca for _, _, _, forca in ativacoes]
    cons    = [r.consequente for r, *_ in ativacoes]
    cores_b = [COR_CONS[c] for c in cons]

    bars = ax_b.bar(range(9), forcas, color=cores_b, edgecolor="#333", lw=0.7, width=0.6)
    ax_b.set_xticks(range(9)); ax_b.set_xticklabels(rotulos, fontsize=8)
    for bar, forca, c in zip(bars, forcas, cons):
        if forca > 0:
            ax_b.text(bar.get_x() + bar.get_width() / 2, forca + 0.02,
                      f"α={forca:.2f}", ha="center", va="bottom",
                      fontsize=8, fontweight="bold")
    leg = [Patch(color=c, label=l) for l, c in COR_CONS.items()]
    ax_b.legend(handles=leg, fontsize=8, loc="upper right",
                title="Consequente AC", title_fontsize=8)
    ax_b.set_ylim(0, 1.2)
    ax_b.set_ylabel("Força de ativação (α)", fontsize=10)
    ax_b.set_title(f"Ativação das 9 Regras  (T={T}°C, U={U}%)",
                   fontsize=11, fontweight="bold")
    ax_b.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Sistema Fuzzy — Regras SE-ENTÃO  (Temperatura × Umidade → Potência AC)",
        fontsize=13, fontweight="bold"
    )

    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Gráfico salvo em {saida}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # BLOCO 3 — ENTRADAS DE TESTE: T=27°C e U=60% do slide 9.
    # Para outro problema, substitua pelos valores de entrada do seu domínio.
    T, U = 27, 60
    graus_t, graus_u, ativacoes = relatorio(T, U)

    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    raiz = os.path.abspath(os.path.join(base, "..", "..", "images"))
    if not os.path.isdir(raiz):
        raiz = os.getcwd()
    saida = os.path.join(raiz, "GO1104.png")
    grafico(T, U, graus_t, graus_u, ativacoes, saida)
