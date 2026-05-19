# =============================================================================
# Identificador: GO1126-Tutorial02-VariaveisEPertinencia
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 2 DE 7 — VARIAVEIS LINGUISTICAS E FUNCOES DE PERTINENCIA
#
# Na Parte 1 vimos que limites rigidos nao funcionam bem. Agora vamos
# criar as FUNCOES DE PERTINENCIA — o coracao do sistema fuzzy.
#
# Uma funcao de pertinencia responde: "com que GRAU este valor pertence
# ao conceito linguistico X?" Por exemplo:
#   pH = 7.0 pertence ao conceito "Neutro" com grau 1.0 (100%)
#   pH = 6.5 pertence ao conceito "Neutro" com grau 0.5 (50%)
#   pH = 5.0 pertence ao conceito "Neutro" com grau 0.0 (0%)
#
# Nesta parte voce vai:
#   1. Entender as funcoes triangular e trapezoidal
#   2. Criar as MFs para pH, Turbidez, OD e Temperatura
#   3. Criar a MF para a saida (Indice de Qualidade)
#   4. Visualizar e testar cada funcao
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt
try:
    get_ipython()
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    pass

import numpy as np


# =============================================================================
# FUNCOES DE PERTINENCIA — IMPLEMENTACAO MANUAL
# (depois veremos como o scikit-fuzzy faz isso automaticamente)
# =============================================================================

def trapmf(x_array, a, b, c, d):
    """
    Funcao Trapezoidal definida por 4 pontos: a, b, c, d

    Forma:
          ___________
         /           \
        /             \
    ___/               \___
    a   b           c   d

    - a: inicio da subida (grau 0 -> 1 comeca aqui)
    - b: pico esquerdo (grau 1 comeca aqui)
    - c: pico direito (grau 1 termina aqui)
    - d: fim da descida (grau 1 -> 0 termina aqui)

    Caso especial: a==b => lado esquerdo vertical (sem rampa de subida)
                   c==d => lado direito vertical (sem rampa de descida)
    """
    result = np.zeros_like(x_array, dtype=float)
    for i, x in enumerate(x_array):
        if x <= a or x >= d:
            result[i] = 0.0
        elif x <= b:
            result[i] = (x - a) / (b - a) if b != a else 1.0
        elif x <= c:
            result[i] = 1.0
        else:
            result[i] = (d - x) / (d - c) if d != c else 1.0
    return result


def pertinencia_ponto(x, a, b, c, d):
    """Calcula a pertinencia de um unico ponto."""
    return float(trapmf(np.array([x]), a, b, c, d)[0])


# =============================================================================
# DEFINICAO DOS CONJUNTOS FUZZY
# =============================================================================

# --- pH [4.0, 10.0] ---
PH_ACIDO    = (4.0, 4.0,  5.5, 6.5)   # trapezoidal aberto a esquerda
PH_NEUTRO   = (6.0, 7.0,  7.5, 8.5)   # triangular/trapezoidal
PH_ALCALINO = (7.5, 9.0, 10.0, 10.0)  # trapezoidal aberto a direita

# --- Turbidez [0, 200] ---
TURB_CLARA    = (  0,   0,  20,  50)
TURB_MODERADA = ( 30,  60,  80, 120)
TURB_TURVA    = ( 90, 140, 200, 200)

# --- Oxigenio Dissolvido [0, 14] ---
OD_BAIXO  = ( 0,  0,  3,  5)
OD_MEDIO  = ( 4,  6,  7,  9)
OD_ALTO   = ( 8, 10, 14, 14)

# --- Temperatura [0, 40] ---
TEMP_FRIA   = ( 0,  0, 10, 18)
TEMP_IDEAL  = (15, 20, 24, 28)
TEMP_QUENTE = (25, 32, 40, 40)

# --- Saida: Indice de Qualidade [0, 100] ---
IQ_CRITICA  = (  0,  0, 10, 20)
IQ_RUIM     = ( 15, 25, 30, 40)
IQ_REGULAR  = ( 35, 45, 55, 65)
IQ_BOA      = ( 60, 70, 75, 82)
IQ_OTIMA    = ( 78, 88, 100, 100)


def testar_pertinencias():
    """Testa as funcoes com valores concretos e explica o resultado."""
    testes = [
        ('pH = 4.5 em ACIDO',    PH_ACIDO,    4.5),
        ('pH = 6.2 em NEUTRO',   PH_NEUTRO,   6.2),
        ('pH = 7.0 em NEUTRO',   PH_NEUTRO,   7.0),
        ('pH = 8.0 em ALCALINO', PH_ALCALINO, 8.0),
        ('Turb = 35 em CLARA',   TURB_CLARA,  35),
        ('Turb = 65 em MODERADA',TURB_MODERADA,65),
        ('OD = 6.0 em MEDIO',    OD_MEDIO,    6.0),
        ('OD = 9.0 em ALTO',     OD_ALTO,     9.0),
    ]

    print("TESTES DE PERTINENCIA:")
    print("-" * 55)
    print(f"  {'Descricao':<30} {'Grau':>8}  {'Barra'}")
    print("-" * 55)
    for desc, params, valor in testes:
        grau = pertinencia_ponto(valor, *params)
        barra = '#' * int(grau * 20)
        print(f"  {desc:<30} {grau:>8.3f}  |{barra}")
    print()


def visualizar_ph():
    """Visualiza os conjuntos fuzzy do pH."""
    x = np.linspace(4, 10, 500)

    mu_acido    = trapmf(x, *PH_ACIDO)
    mu_neutro   = trapmf(x, *PH_NEUTRO)
    mu_alcalino = trapmf(x, *PH_ALCALINO)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor('#f8f9fa')

    ax.fill_between(x, mu_acido,    alpha=0.2, color='#ef5350')
    ax.fill_between(x, mu_neutro,   alpha=0.2, color='#42a5f5')
    ax.fill_between(x, mu_alcalino, alpha=0.2, color='#ab47bc')

    ax.plot(x, mu_acido,    lw=2.5, color='#ef5350', label='Acido')
    ax.plot(x, mu_neutro,   lw=2.5, color='#42a5f5', label='Neutro')
    ax.plot(x, mu_alcalino, lw=2.5, color='#ab47bc', label='Alcalino')

    # Destacar pH = 7.2 (Manancial A)
    ph_exemplo = 7.2
    grau_neutro   = pertinencia_ponto(ph_exemplo, *PH_NEUTRO)
    grau_alcalino = pertinencia_ponto(ph_exemplo, *PH_ALCALINO)
    ax.axvline(ph_exemplo, color='gray', lw=1.5, ls='--', label=f'pH={ph_exemplo}')
    ax.plot([ph_exemplo], [grau_neutro], 'o', color='#42a5f5', markersize=10)
    ax.text(ph_exemplo + 0.05, grau_neutro + 0.04,
            f'Neutro={grau_neutro:.2f}', fontsize=9, color='#42a5f5')

    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axhline(1, color='gray', lw=0.8, ls='--')
    ax.set_xlim(4, 10)
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlabel('pH', fontsize=11)
    ax.set_ylabel('Grau de pertinencia', fontsize=11)
    ax.set_title('Conjuntos Fuzzy — pH da Agua', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualizar_todas():
    """Visualiza todos os conjuntos fuzzy em um grid."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Todos os Conjuntos Fuzzy do Sistema', fontsize=14, fontweight='bold')

    configs = [
        (axes[0, 0], 'pH', np.linspace(4, 10, 500), [
            ('Acido', PH_ACIDO, '#ef5350'),
            ('Neutro', PH_NEUTRO, '#42a5f5'),
            ('Alcalino', PH_ALCALINO, '#ab47bc'),
        ]),
        (axes[0, 1], 'Turbidez (NTU)', np.linspace(0, 200, 500), [
            ('Clara', TURB_CLARA, '#26c6da'),
            ('Moderada', TURB_MODERADA, '#ffa726'),
            ('Turva', TURB_TURVA, '#ef5350'),
        ]),
        (axes[0, 2], 'OD (mg/L)', np.linspace(0, 14, 500), [
            ('Baixo', OD_BAIXO, '#ef5350'),
            ('Medio', OD_MEDIO, '#ffa726'),
            ('Alto', OD_ALTO, '#66bb6a'),
        ]),
        (axes[1, 0], 'Temperatura (C)', np.linspace(0, 40, 500), [
            ('Fria', TEMP_FRIA, '#42a5f5'),
            ('Ideal', TEMP_IDEAL, '#66bb6a'),
            ('Quente', TEMP_QUENTE, '#ef5350'),
        ]),
        (axes[1, 1], 'Indice de Qualidade (saida)', np.linspace(0, 100, 500), [
            ('Critica', IQ_CRITICA, '#f44336'),
            ('Ruim', IQ_RUIM, '#ff7043'),
            ('Regular', IQ_REGULAR, '#ffa726'),
            ('Boa', IQ_BOA, '#66bb6a'),
            ('Otima', IQ_OTIMA, '#26c6da'),
        ]),
    ]

    for ax, titulo, x, termos in configs:
        ax.set_facecolor('#fafafa')
        for nome, params, cor in termos:
            mu = trapmf(x, *params)
            ax.plot(x, mu, lw=2.2, color=cor, label=nome)
            ax.fill_between(x, mu, alpha=0.15, color=cor)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_ylabel('Pertinencia', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.axhline(0, color='gray', lw=0.6, ls='--')
        ax.axhline(1, color='gray', lw=0.6, ls='--')

    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5,
        'Resumo:\n\n'
        'pH:    3 conjuntos\n'
        'Turb:  3 conjuntos\n'
        'OD:    3 conjuntos\n'
        'Temp:  3 conjuntos\n'
        'Saida: 5 conjuntos\n\n'
        'Total: 17 MFs\n'
        'Regras: ~15',
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#e3f2fd', edgecolor='#1565c0'),
        transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 70)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 2: VARIAVEIS LINGUISTICAS E FUNCOES DE PERTINENCIA")
    print("=" * 70)
    print()
    print("CONJUNTOS DEFINIDOS:")
    print()
    print("  pH (4 a 10):         Acido | Neutro | Alcalino")
    print("  Turbidez (0-200 NTU): Clara | Moderada | Turva")
    print("  OD (0-14 mg/L):      Baixo | Medio | Alto")
    print("  Temperatura (0-40C): Fria  | Ideal | Quente")
    print("  Saida (0-100):       Critica|Ruim|Regular|Boa|Otima")
    print()
    testar_pertinencias()
    visualizar_ph()
    visualizar_todas()
    print("PROXIMO PASSO: Parte 3 — Regras e Inferencia")
    print("  -> GO1127-Tutorial03-RegrasEInferencia.py")
    print()
