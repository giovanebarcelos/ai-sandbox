# GO1113-ImplementacaoConceitual
# =============================================================================
# LÓGICA FUZZY TIPO-2 — Interval Type-2 Fuzzy Sets (IT2FS)
# =============================================================================
#
# CONCEITO CENTRAL
# ─────────────────
# Em Fuzzy Tipo-1 cada elemento x tem um grau de pertinência μ(x) FIXO.
# Em Fuzzy Tipo-2 esse grau é incerto: existe uma FAIXA [μ_inf(x), μ_sup(x)]
# chamada FOU (Footprint of Uncertainty).
#
#   μ_inf(x) → MF Inferior: pertinência mínima aceita pelos especialistas
#   μ_sup(x) → MF Superior: pertinência máxima aceita pelos especialistas
#   FOU      → região entre as duas curvas (quanto maior, mais incerteza)
#
# QUANDO USAR TIPO-2?
# ────────────────────
# • Dados muito ruidosos ou com alta variabilidade
# • Especialistas com opiniões divergentes sobre os limites dos termos
# • Ambientes incertos onde uma MF fixa seria imprecisa
# • Quando Tipo-1 não captura bem a incerteza do domínio
#
# ESTE EXEMPLO
# ─────────────
# Variável linguística TEMPERATURA com três termos: Frio, Ameno, Quente.
# Cada termo tem MF inferior e superior triangulares, definidas por
# opiniões levemente diferentes de dois grupos de especialistas.
#
# BIBLIOTECA
# ───────────
# Usa 'type2fuzzy' — instalada automaticamente pela célula abaixo.
# Compatível com Google Colab e ambientes locais.
#
# ═════════════════════════════════════════════════════════════════════════════
# PARA ADAPTAR PARA OUTRO DOMÍNIO — leia os comentários marcados com [ADAPTAR]
# ═════════════════════════════════════════════════════════════════════════════


# ── Instalação automática ─────────────────────────────────────────────────────
# Instala 'type2fuzzy' usando o mesmo interpretador Python que está rodando
# este script (funciona no Colab, Jupyter e ambientes locais sem precisar
# abrir terminal separado).
import subprocess
import sys

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'type2fuzzy', '-q'],
    stdout=subprocess.DEVNULL,   # suprime saída do pip para não poluir o log
    stderr=subprocess.DEVNULL,
)


# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from type2fuzzy import IntervalType2FuzzySet
# IntervalType2FuzzySet: estrutura que armazena cada ponto x como um par
# [lower_membership, upper_membership], representando o intervalo de
# incerteza naquele ponto.


# ── Funções auxiliares ────────────────────────────────────────────────────────

def trimf(x, a, b, c):
    """
    Função de pertinência TRIANGULAR normalizada em [0, 1].

    Parâmetros
    ──────────
    x : array   → universo de discurso (eixo horizontal)
    a : float   → pé esquerdo  (pertinência = 0)
    b : float   → pico central (pertinência = 1)
    c : float   → pé direito   (pertinência = 0)

    [ADAPTAR] Para usar forma trapezoidal ou gaussiana, substitua esta
    função por trapzmf(x, a, b, c, d) ou gaussmf(x, mean, sigma).
    """
    return np.maximum(
        0,
        np.minimum(
            (x - a) / max(b - a, 1e-10),   # rampa crescente: a → b
            (c - x) / max(c - b, 1e-10),   # rampa decrescente: b → c
        ),
    )


def build_it2fs(x, lower_params, upper_params):
    """
    Constrói um IntervalType2FuzzySet ponto a ponto a partir de
    parâmetros triangulares da MF inferior e da MF superior.

    Processo
    ────────
    1. Calcula lower_mf e upper_mf usando trimf()
    2. Para cada ponto xi onde ao menos uma das MFs é > 0, registra
       o par (lower, upper) no objeto IntervalType2FuzzySet.
    3. Retorna o objeto IT2FS e os arrays das duas MFs (para plotagem).

    Parâmetros
    ──────────
    x             : array   → universo de discurso
    lower_params  : tuple   → (a, b, c) da MF inferior
    upper_params  : tuple   → (a, b, c) da MF superior

    Restrição: lower_mf(x) ≤ upper_mf(x) para todo x.
    Violá-la gera um IT2FS inválido.

    [ADAPTAR] Se usar outra forma de MF (trapezoidal, gaussiana),
    substitua as chamadas a trimf() aqui.
    """
    lower_mf = trimf(x, *lower_params)
    upper_mf = trimf(x, *upper_params)

    it2fs = IntervalType2FuzzySet()
    for xi, lo, hi in zip(x, lower_mf, upper_mf):
        if hi > 0 or lo > 0:                         # ignora pontos com μ = 0
            it2fs.add_element_from_values(
                round(float(xi), 4),                 # valor do universo
                round(float(lo), 4),                 # μ inferior
                round(float(hi), 4),                 # μ superior
            )
    return it2fs, lower_mf, upper_mf


# ── [ADAPTAR] Definição dos termos linguísticos ───────────────────────────────
#
# Cada entrada do dicionário representa UM TERMO LINGUÍSTICO da variável.
#
# Estrutura de cada termo:
#   'lower' : (a, b, c)   → parâmetros triangulares da MF INFERIOR
#                           (especialistas mais conservadores / restritivos)
#   'upper' : (a, b, c)   → parâmetros triangulares da MF SUPERIOR
#                           (especialistas mais permissivos / abrangentes)
#   'color' : hex string  → cor para o gráfico
#
# Regra:  upper.a ≤ lower.a  e  lower.c ≤ upper.c
# (a MF superior sempre envolve a inferior — é mais "larga")
#
# [ADAPTAR] Para outro domínio:
#   1. Renomeie as chaves ('Frio', 'Ameno', 'Quente') com seus termos
#   2. Ajuste os valores (a, b, c) para o universo do seu problema
#   3. Defina quantos termos forem necessários (mínimo 2, sem limite)
#   4. Escolha cores distintas para facilitar a leitura visual
#
# Exemplos de outros domínios:
#   Pressão:    'Baixa', 'Normal', 'Alta'       → universo [0, 300] mmHg
#   Qualidade:  'Ruim',  'Regular', 'Boa'       → universo [0, 10]
#   Diagnóstico:'Saudável', 'Risco', 'Doente'   → universo [0, 100] (score)
# ─────────────────────────────────────────────────────────────────────────────

TERMS = {
    # termo      MF inferior (conservador)   MF superior (permissivo)    cor
    'Frio':   {'lower': (0,  5,  15),      'upper': (0,  5,  18),      'color': '#2196F3'},
    'Ameno':  {'lower': (12, 20, 28),      'upper': (10, 20, 30),      'color': '#4CAF50'},
    'Quente': {'lower': (22, 30, 40),      'upper': (20, 30, 40),      'color': '#F44336'},
}

# [ADAPTAR] Universo de discurso — deve cobrir todos os valores possíveis
# da variável de entrada. Ajuste o início, fim e resolução (300 pontos é
# suficiente para a maioria dos domínios; use mais pontos para curvas
# muito estreitas ou domínios muito grandes).
X_START = 0    # valor mínimo do universo   [ADAPTAR]
X_END   = 42   # valor máximo do universo   [ADAPTAR]
X_LABEL = 'Temperatura (°C)'               # [ADAPTAR] rótulo do eixo x
TITLE   = 'Lógica Fuzzy Tipo-2 — Variável "Temperatura"'   # [ADAPTAR]


# ── Execução principal ────────────────────────────────────────────────────────

if __name__ == '__main__':

    # 1. Gera o universo de discurso com resolução fina
    x = np.linspace(X_START, X_END, 300)

    # 2. Constrói os IT2FS para todos os termos
    #    sets_data[nome] = {lower, upper, color, it2fs, lo (array MF inf),
    #                                                   hi (array MF sup)}
    sets_data = {}
    for name, cfg in TERMS.items():
        it2fs, lo_mf, hi_mf = build_it2fs(x, cfg['lower'], cfg['upper'])
        sets_data[name] = {**cfg, 'it2fs': it2fs, 'lo': lo_mf, 'hi': hi_mf}

    # ── Layout do gráfico ─────────────────────────────────────────────────────
    # GridSpec 2×3:
    #   Linha 0, colunas 0-2 (span total) → Painel A: visão geral
    #   Linha 1, coluna 0                 → Painel B: detalhe termo 1
    #   Linha 1, coluna 1                 → Painel C: detalhe termo 2
    #   Linha 1, coluna 2                 → Painel D: detalhe termo 3
    #
    # [ADAPTAR] Se tiver mais de 3 termos, aumente as colunas do GridSpec
    # ou adicione uma terceira linha.
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f'{TITLE}\n'
        'O FOU (Footprint of Uncertainty) captura a divergência entre especialistas',
        fontsize=13, fontweight='bold', y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Painel A: todos os termos sobrepostos ─────────────────────────────────
    # Mostra a variável linguística completa.
    # A linha sólida é a MF Superior; a tracejada é a MF Inferior.
    # O sombreado entre elas é o FOU de cada termo.
    ax_all = fig.add_subplot(gs[0, :])   # ocupa toda a primeira linha
    for name, d in sets_data.items():
        ax_all.fill_between(x, d['lo'], d['hi'],
                            alpha=0.25, color=d['color'])
        ax_all.plot(x, d['hi'], color=d['color'],
                    linewidth=2.2, label=name)
        ax_all.plot(x, d['lo'], '--', color=d['color'],
                    linewidth=1.5, alpha=0.8)

    ax_all.set_title(
        f'Todos os termos — MF Superior (─) e MF Inferior (- -) com FOU sombreado',
        fontsize=10,
    )
    ax_all.set_xlabel(X_LABEL)
    ax_all.set_ylabel('Pertinência μ')
    ax_all.set_xlim(X_START, X_END)
    ax_all.set_ylim(-0.05, 1.15)
    ax_all.legend(fontsize=10)
    ax_all.grid(True, alpha=0.3)

    # ── Painéis B-D: detalhe individual por termo ─────────────────────────────
    # Para cada termo mostra:
    #   • FOU sombreado
    #   • MF Superior (linha sólida)
    #   • MF Inferior (linha tracejada)
    #   • Linha vertical no pico (valor de máxima pertinência)
    #   • Área do FOU = medida numérica da incerteza do especialista
    #     (quanto maior, mais divergência houve na definição do termo)
    for col, (name, d) in enumerate(sets_data.items()):
        ax = fig.add_subplot(gs[1, col])
        lo, hi, color = d['lo'], d['hi'], d['color']

        ax.fill_between(x, lo, hi, alpha=0.35, color=color, label='FOU (incerteza)')
        ax.plot(x, hi, color=color, linewidth=2.5, label='MF Superior (permissivo)')
        ax.plot(x, lo, '--', color=color, linewidth=2,  label='MF Inferior (conservador)')

        # Marca o pico (valor central da MF superior, parâmetro b)
        peak = d['upper'][1]   # [ADAPTAR] se usar gaussmf, o pico é 'mean'
        ax.axvline(peak, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.annotate(
            f'Pico\n{peak}',
            xy=(peak, 1.03), ha='center', fontsize=8, color='gray',
        )

        # Área do FOU: integral de (μ_sup − μ_inf) sobre o universo.
        # Reflete quanto os especialistas divergem para esse termo.
        # Valor = 0 → sem incerteza (Tipo-1 equivalente).
        fou_area = np.trapz(hi - lo, x)
        ax.set_title(f'"{name}"  |  área FOU = {fou_area:.1f}', fontsize=9)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel('Pertinência μ')
        ax.set_xlim(X_START, X_END)
        ax.set_ylim(-0.05, 1.2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Salva antes de exibir (no Colab o plt.show() limpa a figura)
    plt.savefig('GO1113_fuzzy_tipo2.png', dpi=120, bbox_inches='tight')
    plt.show()

    # ── Tabela de pertinências ────────────────────────────────────────────────
    # Para cada valor do universo amostrado mostra μ_inf e μ_sup de cada termo.
    # Útil para verificar manualmente os valores e depurar o modelo.
    #
    # [ADAPTAR] Altere a lista de temperaturas/valores abaixo para os
    # pontos relevantes do seu domínio.
    CHECK_POINTS = [5, 10, 15, 20, 25, 30, 35]   # [ADAPTAR]

    header = f"{'Valor':>7} | " + " | ".join(
        f"{n:>10} inf  {n:>10} sup" for n in TERMS
    )
    print('\n📊 Pertinência em pontos-chave  (μ_inf  /  μ_sup):')
    print('─' * len(header))
    print(header)
    print('─' * len(header))
    for val in CHECK_POINTS:
        idx_x = np.argmin(np.abs(x - val))
        row = f'{val:>6}  |'
        for d in sets_data.values():
            row += f"  {d['lo'][idx_x]:>6.2f}        {d['hi'][idx_x]:>6.2f}  |"
        print(row)

    print(f'\n✅ Gráfico salvo em: GO1113_fuzzy_tipo2.png')


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
