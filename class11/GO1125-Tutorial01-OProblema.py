# =============================================================================
# Identificador: GO1125-Tutorial01-OProblema
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 1 DE 7 — O PROBLEMA
#
# Imagine que voce trabalha em uma estacao de tratamento de agua. Todo dia,
# centenas de amostras chegam dos rios e reservatorios. Tecnicos precisam
# decidir rapidamente se a agua e segura, moderada ou critica — mas as
# medidas nunca sao "perfeitas":
#
#   pH = 7.1 e bom? E 6.9? E 7.5? Onde esta o limite exato?
#   Turbidez de 45 NTU e aceitavel? E 46? Qual e a fronteira?
#
# A Logica Fuzzy resolve exatamente esse problema: em vez de linhas rigidas
# (bom / ruim), ela trabalha com GRAUS de qualidade — exatamente como um
# especialista humano raciocina.
#
# Nesta parte voce vai:
#   1. Entender por que Logica Fuzzy e ideal para monitoramento ambiental
#   2. Conhecer as 4 variaveis do nosso sistema
#   3. Ver exemplos de amostras reais (simuladas)
#   4. Entender o fluxo completo: sensores -> fuzzy -> indice -> acao
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
# AS 4 VARIAVEIS DO SISTEMA
# =============================================================================

# BLOCO 1 — VARIÁVEIS DO SISTEMA: define as 4 grandezas físico-químicas monitoradas.
# Para outro problema de monitoramento ambiental: substitua pelas variáveis
# relevantes (ex: nitrogênio, coliformes, condutividade). Cada variável precisa
# de um universo de discurso e de uma faixa 'ideal' para calibrar as MFs.
VARIAVEIS = {
    'pH': {
        'unidade': 'pH',
        'universo': (4.0, 10.0),
        'ideal': (6.5, 8.5),
        'descricao': 'Acidez ou alcalinidade da agua',
    },
    'Turbidez': {
        'unidade': 'NTU',
        'universo': (0, 200),
        'ideal': (0, 40),
        'descricao': 'Clareza visual — quanto a agua esta "suja"',
    },
    'OD': {
        'unidade': 'mg/L',
        'universo': (0, 14),
        'ideal': (7, 14),
        'descricao': 'Oxigenio dissolvido — vital para organismos aquaticos',
    },
    'Temperatura': {
        'unidade': 'C',
        'universo': (0, 40),
        'ideal': (15, 28),
        'descricao': 'Temperatura da agua',
    },
}

# BLOCO 2 — AMOSTRAS: 5 pontos de coleta que cobrem o espectro de qualidade
# (de preservado a industrial). Para outro problema: substitua pelos seus dados
# reais. Estas amostras são usadas para demonstrar as limitações da lógica clássica.
# Amostras simuladas de diferentes pontos de coleta
AMOSTRAS = [
    {'nome': 'Manancial A (preservado)', 'ph': 7.2, 'turb': 8,   'od': 9.5, 'temp': 20},
    {'nome': 'Rio B (zona urbana)',      'ph': 6.8, 'turb': 55,  'od': 6.2, 'temp': 24},
    {'nome': 'Lago C (eutrofizado)',     'ph': 5.5, 'turb': 130, 'od': 2.8, 'temp': 31},
    {'nome': 'Reservatorio D',          'ph': 8.1, 'turb': 22,  'od': 8.0, 'temp': 18},
    {'nome': 'Efluente E (industrial)', 'ph': 4.2, 'turb': 180, 'od': 1.2, 'temp': 38},
]


def exibir_tabela_amostras():
    print("=" * 70)
    print("  AMOSTRAS DE AGUA — DIFERENTES PONTOS DE COLETA")
    print("=" * 70)
    print(f"  {'Ponto':<28} {'pH':>5} {'Turb':>8} {'OD':>7} {'Temp':>7}")
    print(f"  {'':28} {'':>5} {'(NTU)':>8} {'(mg/L)':>7} {'(C)':>7}")
    print("-" * 70)
    for a in AMOSTRAS:
        print(f"  {a['nome']:<28} {a['ph']:>5.1f} {a['turb']:>8.0f} "
              f"{a['od']:>7.1f} {a['temp']:>7.0f}")
    print("=" * 70)


def problema_da_logica_classica():
    """Mostra por que limites rigidos nao funcionam bem."""
    print()
    print("PROBLEMA COM LOGICA CLASSICA (limites rigidos):")
    print("-" * 50)

    # LÓGICA CLÁSSICA: limites binários rígidos. Rio B (pH=6.8) é classificado como RUIM
    # mesmo estando a 0.3 unidades do limite — é exatamente este problema que a fuzzy resolve.
    # Regra classica: pH bom se entre 6.5 e 8.5
    limites = {'ph': (6.5, 8.5), 'turb': (0, 40), 'od': (7.0, 14)}

    for a in AMOSTRAS:
        ph_ok   = limites['ph'][0]  <= a['ph']   <= limites['ph'][1]
        turb_ok = limites['turb'][0] <= a['turb'] <= limites['turb'][1]
        od_ok   = limites['od'][0]  <= a['od']   <= limites['od'][1]

        todos_ok = ph_ok and turb_ok and od_ok
        status = "BOA" if todos_ok else "RUIM"

        print(f"  {a['nome'][:25]:<25}: {status}")
        if not todos_ok:
            problemas = []
            if not ph_ok:
                problemas.append(f"pH={a['ph']}")
            if not turb_ok:
                problemas.append(f"Turb={a['turb']}")
            if not od_ok:
                problemas.append(f"OD={a['od']}")
            print(f"    Problema: {', '.join(problemas)}")

    print()
    print("PROBLEMA: Rio B tem pH=6.8 (muito proximo de 6.5) e e classificado")
    print("como RUIM — mas na pratica e agua de qualidade aceitavel!")
    print()
    print("SOLUCAO: Logica Fuzzy usa GRAUS de qualidade, nao fronteiras rigidas.")


def visualizar_variaveis():
    """Grafico das 4 variaveis mostrando os intervalos das amostras."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('Variaveis de Qualidade da Agua — Amostras Coletadas',
                 fontsize=14, fontweight='bold')

    configs = [
        (axes[0, 0], 'pH', 'ph', (4, 10), (6.5, 8.5)),
        (axes[0, 1], 'Turbidez (NTU)', 'turb', (0, 200), (0, 40)),
        (axes[1, 0], 'Oxigenio Dissolvido (mg/L)', 'od', (0, 14), (7, 14)),
        (axes[1, 1], 'Temperatura (C)', 'temp', (0, 40), (15, 28)),
    ]

    cores = ['#26c6da', '#66bb6a', '#ffa726', '#ef5350', '#ab47bc']

    for ax, titulo, chave, (xmin, xmax), (ideal_min, ideal_max) in configs:
        ax.set_facecolor('#f8f9fa')

        # Zona ideal
        ax.axvspan(ideal_min, ideal_max, alpha=0.15, color='green',
                   label='Zona ideal')
        ax.axvline(ideal_min, color='green', lw=1, ls='--', alpha=0.5)
        ax.axvline(ideal_max, color='green', lw=1, ls='--', alpha=0.5)

        # Pontos das amostras
        for i, (amostra, cor) in enumerate(zip(AMOSTRAS, cores)):
            valor = amostra[chave]
            ypos = 0.5 + i * 0.1
            ax.plot(valor, ypos, 'o', color=cor, markersize=10, zorder=5)
            ax.text(valor, ypos + 0.06, f"  {amostra['nome'][:12]}",
                    fontsize=7, color=cor, va='bottom')

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_xlabel(titulo, fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.show()
    print("\nGrafico gerado: distribuicao das amostras nas 4 variaveis")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 70)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 1: O PROBLEMA")
    print("=" * 70)
    print()
    print("  Objetivo: classificar a qualidade da agua de forma inteligente")
    print("  usando 4 parametros fisico-quimicos e Logica Fuzzy.")
    print()

    print("VARIAVEIS DO SISTEMA:")
    print("-" * 50)
    for nome, info in VARIAVEIS.items():
        print(f"  {nome:<12}: {info['descricao']}")
        print(f"              Universo: {info['universo']} {info['unidade']}")
        print(f"              Ideal:    {info['ideal']} {info['unidade']}")
    print()

    exibir_tabela_amostras()
    problema_da_logica_classica()
    visualizar_variaveis()

    print()
    print("PROXIMO PASSO: Parte 2 — Funcoes de Pertinencia")
    print("  -> GO1126-Tutorial02-VariaveisEPertinencia.py")
    print()
