# GO1105-ImplementacaoPythonSimplificada
# Inferência de Mamdani — Sistema de Gorjeta (Slide 10)
#
# Entradas validadas contra o slide (Serviço=6.5, Comida=7.0):
#   μ_SERV_RUIM=0.0  μ_SERV_MÉDIO=0.5  μ_SERV_BOM=0.5
#   μ_COMIDA_RUIM=0.0  μ_COMIDA_MÉDIA=0.4  μ_COMIDA_BOA=0.6
#   R1(OR)=0.0→BAIXA  R2(AND)=0.4→MÉDIA  R3(AND)=0.5→ALTA
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Funções de pertinência ────────────────────────────────────────────────────

def mf_trap(x, a, b, c, d):
    """Trapezoidal: sobe a→b, plano b→c, desce c→d. Aceita escalar ou array."""
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.zeros_like(arr)
    if b > a:
        m = (arr > a) & (arr < b)
        y[m] = (arr[m] - a) / (b - a)
    y[(arr >= b) & (arr <= c)] = 1.0
    if d > c:
        m = (arr > c) & (arr < d)
        y[m] = (d - arr[m]) / (d - c)
    return float(y[0]) if np.ndim(x) == 0 else y


def mf_tri(x, a, b, c):
    return mf_trap(x, a, b, b, c)


# Serviço [0, 10]: RUIM(6.5)=0.0  MÉDIO(6.5)=0.5  BOM(6.5)=0.5
FUNCS_SERV = {
    "RUIM":  lambda x: mf_trap(x, 0,  0,  2,  5),
    "MÉDIO": lambda x: mf_tri (x, 2,  5,  8),
    "BOM":   lambda x: mf_trap(x, 5,  8, 10, 10),
}

# Comida [0, 10]: RUIM(7.0)=0.0  MÉDIA(7.0)=0.4  BOA(7.0)=0.6
FUNCS_COMIDA = {
    "RUIM":  lambda x: mf_trap(x, 0,  0,  1,  4),
    "MÉDIA": lambda x: mf_tri (x, 1,  4,  9),
    "BOA":   lambda x: mf_trap(x, 4,  9, 10, 10),
}

# Gorjeta [0, 25%]
FUNCS_GORJETA = {
    "BAIXA": lambda x: mf_tri(x,  0,  0, 13),
    "MÉDIA": lambda x: mf_tri(x,  0, 13, 25),
    "ALTA":  lambda x: mf_tri(x, 13, 25, 25),
}

COR = {"BAIXA": "#3498DB", "MÉDIA": "#2ECC71", "ALTA": "#E74C3C"}


# ── Regras ────────────────────────────────────────────────────────────────────

class Regra:
    def __init__(self, numero, op, ant_serv, ant_comida, consequente):
        self.numero = numero
        self.op = op                  # "AND" | "OR"
        self.ant_serv = ant_serv      # termo do serviço
        self.ant_comida = ant_comida  # termo da comida
        self.consequente = consequente

    def avaliar(self, gs, gc):
        a, b = gs[self.ant_serv], gc[self.ant_comida]
        return max(a, b) if self.op == "OR" else min(a, b)

    def descricao(self):
        return (f"R{self.numero}: SE serv={self.ant_serv} "
                f"{self.op} comida={self.ant_comida} → {self.consequente}")


REGRAS = [
    Regra(1, "OR",  "RUIM",  "RUIM",  "BAIXA"),
    Regra(2, "AND", "MÉDIO", "MÉDIA", "MÉDIA"),
    Regra(3, "AND", "BOM",   "BOA",   "ALTA"),
]


# ── Inferência ────────────────────────────────────────────────────────────────

def mamdani(servico, comida):
    x = np.linspace(0, 25, 500)

    gs = {t: f(servico) for t, f in FUNCS_SERV.items()}
    gc = {t: f(comida)  for t, f in FUNCS_COMIDA.items()}

    forcas, detalhes = {}, []
    for r in REGRAS:
        alpha = r.avaliar(gs, gc)
        forcas[r.consequente] = max(forcas.get(r.consequente, 0.0), alpha)
        detalhes.append((r, alpha))

    clips = {t: np.minimum(FUNCS_GORJETA[t](x), forcas[t]) for t in forcas}
    agregado = np.zeros_like(x)
    for clip in clips.values():
        agregado = np.maximum(agregado, clip)

    denom = float(np.sum(agregado))
    centroide = float(np.sum(agregado * x) / denom) if denom > 0 else 0.0

    return gs, gc, forcas, detalhes, clips, agregado, centroide, x


def relatorio(servico, comida):
    gs, gc, forcas, detalhes, clips, agregado, centroide, x = mamdani(servico, comida)

    print("=" * 58)
    print("  INFERÊNCIA DE MAMDANI — SISTEMA DE GORJETA")
    print("=" * 58)
    print(f"  Entradas: Serviço = {servico}  |  Comida = {comida}\n")

    print("── 1. Fuzzificação ─────────────────────────────────────")
    for t, mu in gs.items():
        print(f"  μ_SERVIÇO_{t:<5}({servico}) = {mu:.2f}  {'█'*int(mu*20)}")
    print()
    for t, mu in gc.items():
        print(f"  μ_COMIDA_{t:<5}({comida})  = {mu:.2f}  {'█'*int(mu*20)}")
    print()

    print("── 2. Inferência ───────────────────────────────────────")
    for r, alpha in detalhes:
        a = gs[r.ant_serv]; b = gc[r.ant_comida]
        op = "max" if r.op == "OR" else "min"
        print(f"  R{r.numero}: {op}({a:.1f}, {b:.1f}) = {alpha:.2f}"
              f"  → GORJETA {r.consequente}")
    print()

    print("── 3. Defuzzificação (Centroide) ───────────────────────")
    print(f"  Gorjeta = {centroide:.1f}%")
    print("=" * 58)

    return gs, gc, forcas, detalhes, clips, agregado, centroide, x


# ── Gráfico ───────────────────────────────────────────────────────────────────

def grafico(servico, comida, gs, gc, forcas, detalhes, clips, agregado, centroide, x_out, saida):
    xs = np.linspace(0, 10, 300)

    fig = plt.figure(figsize=(15, 9))
    layout = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.52, wspace=0.38,
                               height_ratios=[1, 1.3])
    ax_s = fig.add_subplot(layout[0, 0])
    ax_c = fig.add_subplot(layout[0, 1])
    ax_r = fig.add_subplot(layout[0, 2])
    ax_o = fig.add_subplot(layout[1, :])

    # ── ① Serviço ──────────────────────────────────────────
    mfs_s = [("RUIM", FUNCS_SERV["RUIM"], "#3498DB"),
             ("MÉDIO", FUNCS_SERV["MÉDIO"], "#2ECC71"),
             ("BOM", FUNCS_SERV["BOM"], "#E74C3C")]
    for nome, func, cor in mfs_s:
        y = func(xs)
        ax_s.plot(xs, y, color=cor, lw=2, label=nome)
        ax_s.fill_between(xs, y, alpha=0.12, color=cor)
    ax_s.axvline(servico, color="purple", lw=1.5, ls="--", label=f"S={servico}")
    for nome, _, cor in mfs_s:
        mu = gs[nome]
        if mu > 0:
            ax_s.scatter(servico, mu, color=cor, edgecolors="#222", s=70, zorder=5)
            ax_s.annotate(f"{mu:.1f}", (servico, mu),
                          xytext=(servico + 0.25, mu + 0.06), fontsize=8, color=cor)
    ax_s.set_xlim(0, 10); ax_s.set_ylim(-0.05, 1.2)
    ax_s.set_xlabel("Qualidade do Serviço [0–10]", fontsize=9)
    ax_s.set_ylabel("μ", fontsize=10)
    ax_s.set_title("① Fuzzificação — Serviço", fontsize=10, fontweight="bold")
    ax_s.legend(fontsize=8); ax_s.grid(axis="y", alpha=0.3)

    # ── ① Comida ───────────────────────────────────────────
    mfs_c = [("RUIM", FUNCS_COMIDA["RUIM"], "#3498DB"),
             ("MÉDIA", FUNCS_COMIDA["MÉDIA"], "#2ECC71"),
             ("BOA", FUNCS_COMIDA["BOA"], "#E74C3C")]
    for nome, func, cor in mfs_c:
        y = func(xs)
        ax_c.plot(xs, y, color=cor, lw=2, label=nome)
        ax_c.fill_between(xs, y, alpha=0.12, color=cor)
    ax_c.axvline(comida, color="purple", lw=1.5, ls="--", label=f"C={comida}")
    for nome, _, cor in mfs_c:
        mu = gc[nome]
        if mu > 0:
            ax_c.scatter(comida, mu, color=cor, edgecolors="#222", s=70, zorder=5)
            ax_c.annotate(f"{mu:.1f}", (comida, mu),
                          xytext=(comida + 0.25, mu + 0.06), fontsize=8, color=cor)
    ax_c.set_xlim(0, 10); ax_c.set_ylim(-0.05, 1.2)
    ax_c.set_xlabel("Qualidade da Comida [0–10]", fontsize=9)
    ax_c.set_ylabel("μ", fontsize=10)
    ax_c.set_title("① Fuzzificação — Comida", fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=8); ax_c.grid(axis="y", alpha=0.3)

    # ── ② Regras ───────────────────────────────────────────
    labels_r = [f"R{r.numero}  {r.op}\n{r.ant_serv[:3]}/{r.ant_comida[:3]}\n→ {r.consequente}"
                for r, _ in detalhes]
    forcas_r = [alpha for _, alpha in detalhes]
    cores_r  = [COR[r.consequente] for r, _ in detalhes]

    bars = ax_r.barh(range(3), forcas_r, color=cores_r,
                     edgecolor="#333", lw=0.7, height=0.45)
    ax_r.set_yticks(range(3))
    ax_r.set_yticklabels(labels_r, fontsize=8)
    ax_r.set_xlim(0, 1.15)
    ax_r.set_xlabel("Força de ativação (α)", fontsize=9)
    ax_r.set_title("② Inferência — Regras", fontsize=10, fontweight="bold")
    for bar, alpha in zip(bars, forcas_r):
        ax_r.text(alpha + 0.02, bar.get_y() + bar.get_height() / 2,
                  f"α={alpha:.2f}", va="center", fontsize=9, fontweight="bold")
    ax_r.axvline(0, color="gray", lw=0.8)
    ax_r.grid(axis="x", alpha=0.3)

    # ── ③ Saída: clips + agregação + centroide ────────────
    # MFs completas (pontilhadas) como referência
    for nome, func in FUNCS_GORJETA.items():
        ax_o.plot(x_out, func(x_out), color=COR[nome], lw=1, ls=":", alpha=0.35)

    # MFs clipadas
    for nome, clip in clips.items():
        alpha_val = forcas[nome]
        ax_o.fill_between(x_out, clip, alpha=0.40, color=COR[nome],
                          label=f"GORJETA {nome}  (clip α={alpha_val:.1f})")
        ax_o.plot(x_out, clip, color=COR[nome], lw=2)

    # Área agregada (máximo)
    ax_o.fill_between(x_out, agregado, alpha=0.12, color="#6C3483",
                      label="Área agregada (máx)")
    ax_o.plot(x_out, agregado, color="#6C3483", lw=2.5, ls="-", alpha=0.7)

    # Centroide
    ax_o.axvline(centroide, color="black", lw=2.5, ls="--",
                 label=f"Centroide = {centroide:.1f}%")
    ax_o.annotate(
        f"  Gorjeta ≈ {centroide:.1f}%",
        xy=(centroide, 0.52),
        xytext=(centroide + 1.8, 0.72),
        fontsize=11, fontweight="bold", color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    ax_o.set_xlim(0, 25); ax_o.set_ylim(-0.05, 1.15)
    ax_o.set_xlabel("Gorjeta (%)", fontsize=11)
    ax_o.set_ylabel("μ", fontsize=11)
    ax_o.set_title(
        "③ Defuzzificação — Centroide (área agregada → valor nítido)",
        fontsize=11, fontweight="bold"
    )
    ax_o.legend(fontsize=9, loc="upper left")
    ax_o.grid(axis="y", alpha=0.3)
    ax_o.set_xticks(range(0, 26, 5))

    fig.suptitle(
        f"Inferência de Mamdani — Sistema de Gorjeta  "
        f"(Serviço={servico}, Comida={comida})",
        fontsize=13, fontweight="bold",
    )

    plt.savefig(saida, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Gráfico salvo em {saida}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SERVICO, COMIDA = 6.5, 7.0

    gs, gc, forcas, detalhes, clips, agregado, centroide, x_out = relatorio(SERVICO, COMIDA)

    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()
    raiz = os.path.abspath(os.path.join(base, "..", "..", "images"))
    if not os.path.isdir(raiz):
        raiz = os.getcwd()
    saida = os.path.join(raiz, "GO1105.png")
    grafico(SERVICO, COMIDA, gs, gc, forcas, detalhes, clips, agregado, centroide, x_out, saida)
