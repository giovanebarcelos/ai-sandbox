# GO1113-ImplementacaoConceitual
# Conjuntos Fuzzy Tipo-2 — modela incerteza nas próprias funções de
# pertinência: em vez de uma curva fixa, a MF tem uma "faixa" (FOU).
# Demonstra variável linguística TEMPERATURA com três termos fuzzy:
# Frio, Ameno e Quente — cada um com MF inferior e superior definidas
# por opiniões divergentes de especialistas.
#
# Requer: type2fuzzy  →  instalado automaticamente abaixo.
# Compatível com Google Colab e ambientes locais.
#
# Para adaptar: ajuste lower_params e upper_params de cada termo para
# refletir a incerteza do seu domínio (ex: diagnóstico médico).

# ── Instalação automática ─────────────────────────────────────────────────────
import subprocess
import sys

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'type2fuzzy', '-q'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from type2fuzzy import IntervalType2FuzzySet


# ── Helpers ───────────────────────────────────────────────────────────────────
def trimf(x, a, b, c):
    """Função de pertinência triangular normalizada em [0, 1]."""
    return np.maximum(
        0,
        np.minimum(
            (x - a) / max(b - a, 1e-10),
            (c - x) / max(c - b, 1e-10),
        ),
    )


def build_it2fs(x, lower_params, upper_params):
    """Constrói um IntervalType2FuzzySet a partir de parâmetros triangulares."""
    lower_mf = trimf(x, *lower_params)
    upper_mf = trimf(x, *upper_params)
    it2fs = IntervalType2FuzzySet()
    for xi, lo, hi in zip(x, lower_mf, upper_mf):
        if hi > 0 or lo > 0:
            it2fs.add_element_from_values(
                round(float(xi), 4),
                round(float(lo), 4),
                round(float(hi), 4),
            )
    return it2fs, lower_mf, upper_mf


# ── Definição dos termos linguísticos ─────────────────────────────────────────
TERMS = {
    'Frio':   {'lower': (0,  5,  15), 'upper': (0,  5,  18), 'color': '#2196F3'},
    'Ameno':  {'lower': (12, 20, 28), 'upper': (10, 20, 30), 'color': '#4CAF50'},
    'Quente': {'lower': (22, 30, 40), 'upper': (20, 30, 40), 'color': '#F44336'},
}

if __name__ == '__main__':
    x = np.linspace(0, 42, 300)

    # Construir todos os IT2FS
    sets_data = {}
    for name, cfg in TERMS.items():
        it2fs, lo_mf, hi_mf = build_it2fs(x, cfg['lower'], cfg['upper'])
        sets_data[name] = {**cfg, 'it2fs': it2fs, 'lo': lo_mf, 'hi': hi_mf}

    # ── Layout: 1 painel geral + 3 detalhes + 1 comparação Type-1 vs Type-2 ──
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        'Lógica Fuzzy Tipo-2 — Variável "Temperatura"\n'
        'O FOU (Footprint of Uncertainty) captura a divergência entre especialistas',
        fontsize=13, fontweight='bold', y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Painel A: visão geral ─────────────────────────────────────────────────
    ax_all = fig.add_subplot(gs[0, :])
    for name, d in sets_data.items():
        ax_all.fill_between(x, d['lo'], d['hi'],
                            alpha=0.25, color=d['color'])
        ax_all.plot(x, d['hi'], color=d['color'],
                    linewidth=2.2, label=f'{name}')
        ax_all.plot(x, d['lo'], '--', color=d['color'],
                    linewidth=1.5, alpha=0.8)
    ax_all.set_title('Todos os termos — MF Superior (–) e Inferior (- -) com FOU sombreado',
                     fontsize=10)
    ax_all.set_xlabel('Temperatura (°C)')
    ax_all.set_ylabel('Pertinência μ')
    ax_all.set_xlim(0, 42)
    ax_all.set_ylim(-0.05, 1.15)
    ax_all.legend(fontsize=10)
    ax_all.grid(True, alpha=0.3)

    # ── Painéis B-D: detalhe por termo ───────────────────────────────────────
    for col, (name, d) in enumerate(sets_data.items()):
        ax = fig.add_subplot(gs[1, col])
        lo, hi, color = d['lo'], d['hi'], d['color']

        ax.fill_between(x, lo, hi, alpha=0.35, color=color, label='FOU')
        ax.plot(x, hi, color=color, linewidth=2.5, label='MF Superior')
        ax.plot(x, lo, '--', color=color, linewidth=2, label='MF Inferior')

        # Largura do FOU no pico
        peak = d['upper'][1]
        ax.axvline(peak, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.annotate(f'Pico\n{peak}°C', xy=(peak, 1.03),
                    ha='center', fontsize=8, color='gray')

        # Área FOU (incerteza)
        fou_area = np.trapz(hi - lo, x)
        ax.set_title(f'"{name}"\n(área FOU = {fou_area:.1f})', fontsize=9)
        ax.set_xlabel('Temperatura (°C)')
        ax.set_ylabel('Pertinência μ')
        ax.set_xlim(0, 42)
        ax.set_ylim(-0.05, 1.2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.savefig('GO1113_fuzzy_tipo2.png', dpi=120, bbox_inches='tight')
    plt.show()

    # ── Tabela de pertinências ────────────────────────────────────────────────
    header = f"{'Temp':>6} | " + " | ".join(
        f"{n+' inf':>10} {n+' sup':>10}" for n in TERMS
    )
    print('\n📊 Pertinência em pontos-chave (inferior / superior):')
    print('-' * len(header))
    print(header)
    print('-' * len(header))
    for temp in [5, 10, 15, 20, 25, 30, 35]:
        idx_x = np.argmin(np.abs(x - temp))
        row = f'{temp:>5}°C |'
        for d in sets_data.values():
            row += f" {d['lo'][idx_x]:>10.2f} {d['hi'][idx_x]:>10.2f} |"
        print(row)
    print(f'\n✅ Gráfico salvo em: GO1113_fuzzy_tipo2.png')
