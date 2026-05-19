# =============================================================================
# Identificador: GO1129-Tutorial05-SistemaCompleto
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 5 DE 7 — SISTEMA COMPLETO COM SCIKIT-FUZZY
#
# Nas partes anteriores implementamos tudo manualmente para entender
# cada etapa. Agora vamos usar a biblioteca SCIKIT-FUZZY, que automatiza
# todo o processo com uma API elegante e eficiente.
#
# Instalacao:
#   pip install scikit-fuzzy matplotlib numpy
#
# O scikit-fuzzy cuida de:
#   - Criacao das variaveis (Antecedent / Consequent)
#   - Definicao das MFs (trimf, trapmf, gaussmf, etc.)
#   - Compilacao das regras (Rule)
#   - Sistema de controle (ControlSystem)
#   - Simulacao e defuzzificacao (ControlSystemSimulation)
#
# Nesta parte voce vai:
#   1. Recriar todo o sistema usando a API do scikit-fuzzy
#   2. Executar simulacoes para todas as amostras
#   3. Comparar os resultados com a implementacao manual
#   4. Explorar opcoes de defuzzificacao da biblioteca
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt
try:
    get_ipython()
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    pass

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    print("ERRO: scikit-fuzzy nao encontrado.")
    print("Instale com: pip install scikit-fuzzy")
    raise


# =============================================================================
# CONSTRUCAO DO SISTEMA FUZZY COM SCIKIT-FUZZY
# =============================================================================

def construir_sistema():
    """Constroi o sistema fuzzy de qualidade da agua com scikit-fuzzy."""

    # --- Universos de discurso ---
    ph_u    = np.arange(4.0, 10.1, 0.05)
    turb_u  = np.arange(0.0, 201.0, 1.0)
    od_u    = np.arange(0.0, 14.1, 0.1)
    temp_u  = np.arange(0.0, 40.1, 0.5)
    qual_u  = np.arange(0.0, 100.1, 0.5)

    # --- Variaveis de entrada (Antecedent) e saida (Consequent) ---
    pH       = ctrl.Antecedent(ph_u,   'pH')
    turbidez = ctrl.Antecedent(turb_u, 'Turbidez')
    od       = ctrl.Antecedent(od_u,   'OD')
    temp     = ctrl.Antecedent(temp_u, 'Temperatura')
    qualidade= ctrl.Consequent(qual_u, 'Qualidade', defuzzify_method='centroid')

    # --- Funcoes de Pertinencia das entradas ---
    pH['Acido']    = fuzz.trapmf(ph_u,   [4.0, 4.0, 5.5, 6.5])
    pH['Neutro']   = fuzz.trapmf(ph_u,   [6.0, 7.0, 7.5, 8.5])
    pH['Alcalino'] = fuzz.trapmf(ph_u,   [7.5, 9.0,10.0,10.0])

    turbidez['Clara']    = fuzz.trapmf(turb_u, [  0,  0, 20,  50])
    turbidez['Moderada'] = fuzz.trapmf(turb_u, [ 30, 60, 80, 120])
    turbidez['Turva']    = fuzz.trapmf(turb_u, [ 90,140,200, 200])

    od['Baixo'] = fuzz.trapmf(od_u, [0, 0,  3,  5])
    od['Medio'] = fuzz.trapmf(od_u, [4, 6,  7,  9])
    od['Alto']  = fuzz.trapmf(od_u, [8,10, 14, 14])

    temp['Fria']   = fuzz.trapmf(temp_u, [ 0,  0, 10, 18])
    temp['Ideal']  = fuzz.trapmf(temp_u, [15, 20, 24, 28])
    temp['Quente'] = fuzz.trapmf(temp_u, [25, 32, 40, 40])

    # --- Funcoes de Pertinencia da saida ---
    qualidade['Critica'] = fuzz.trapmf(qual_u, [ 0,  0, 10, 20])
    qualidade['Ruim']    = fuzz.trapmf(qual_u, [15, 25, 30, 40])
    qualidade['Regular'] = fuzz.trapmf(qual_u, [35, 45, 55, 65])
    qualidade['Boa']     = fuzz.trapmf(qual_u, [60, 70, 75, 82])
    qualidade['Otima']   = fuzz.trapmf(qual_u, [78, 88,100,100])

    # --- Base de Regras ---
    regras = [
        # Otima
        ctrl.Rule(pH['Neutro']   & turbidez['Clara']    & od['Alto'],           qualidade['Otima']),
        ctrl.Rule(pH['Neutro']   & turbidez['Clara']    & od['Medio'] & temp['Ideal'], qualidade['Otima']),
        # Boa
        ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od['Alto'],           qualidade['Boa']),
        ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od['Medio'],          qualidade['Boa']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Clara']    & od['Alto'],           qualidade['Boa']),
        # Regular
        ctrl.Rule(pH['Acido']    & turbidez['Moderada'] & od['Medio'],          qualidade['Regular']),
        ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od['Baixo'],          qualidade['Regular']),
        ctrl.Rule(pH['Neutro']   & turbidez['Turva']    & od['Medio'],          qualidade['Regular']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Moderada'] & od['Medio'],          qualidade['Regular']),
        # Ruim
        ctrl.Rule(pH['Acido']    & turbidez['Turva']    & od['Medio'],          qualidade['Ruim']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Turva']    & od['Medio'],          qualidade['Ruim']),
        ctrl.Rule(pH['Neutro']   & turbidez['Turva']    & od['Baixo'],          qualidade['Ruim']),
        # Critica
        ctrl.Rule(pH['Acido']    & turbidez['Turva']    & od['Baixo'],          qualidade['Critica']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Turva']    & od['Baixo'],          qualidade['Critica']),
        ctrl.Rule(temp['Quente'] & od['Baixo'],                                 qualidade['Critica']),
    ]

    sistema_ctrl = ctrl.ControlSystem(regras)
    simulacao    = ctrl.ControlSystemSimulation(sistema_ctrl)

    return pH, turbidez, od, temp, qualidade, simulacao


def classificar(indice):
    if indice >= 80:
        return 'OTIMA',   '#26c6da'
    elif indice >= 60:
        return 'BOA',     '#66bb6a'
    elif indice >= 40:
        return 'REGULAR', '#ffa726'
    elif indice >= 20:
        return 'RUIM',    '#ff7043'
    else:
        return 'CRITICA', '#f44336'


def avaliar_amostra(simulacao, nome, ph_val, turb_val, od_val, temp_val):
    """Avalia uma amostra e retorna o indice de qualidade."""
    simulacao.input['pH']          = ph_val
    simulacao.input['Turbidez']    = turb_val
    simulacao.input['OD']          = od_val
    simulacao.input['Temperatura'] = temp_val
    simulacao.compute()
    indice = simulacao.output['Qualidade']
    classe, _ = classificar(indice)
    return indice, classe


def visualizar_mfs_skfuzzy(pH, turbidez, od, temp, qualidade):
    """Visualiza as MFs usando o metodo nativo do scikit-fuzzy."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Variaveis Fuzzy — scikit-fuzzy', fontsize=14, fontweight='bold')

    pH.view(ax=axes[0, 0])
    turbidez.view(ax=axes[0, 1])
    od.view(ax=axes[0, 2])
    temp.view(ax=axes[1, 0])
    qualidade.view(ax=axes[1, 1])

    axes[0, 0].set_title('pH', fontsize=12)
    axes[0, 1].set_title('Turbidez (NTU)', fontsize=12)
    axes[0, 2].set_title('OD (mg/L)', fontsize=12)
    axes[1, 0].set_title('Temperatura (C)', fontsize=12)
    axes[1, 1].set_title('Qualidade (saida)', fontsize=12)
    axes[1, 2].axis('off')

    axes[1, 2].text(0.5, 0.5,
        'scikit-fuzzy API:\n\n'
        'ctrl.Antecedent(u, nome)\n'
        'ctrl.Consequent(u, nome)\n'
        'fuzz.trapmf(u, [a,b,c,d])\n'
        'ctrl.Rule(cond, cons)\n'
        'ctrl.ControlSystem(regras)\n'
        'sim.compute()',
        ha='center', va='center', fontsize=10,
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='#2e7d32'),
        transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.show()


def visualizar_resultados(resultados):
    """Grafico de barras horizontais com os resultados."""
    nomes  = [r['nome'][:20] for r in resultados]
    indices = [r['indice'] for r in resultados]
    classes = [r['classe'] for r in resultados]

    cores_classes = {
        'OTIMA': '#26c6da', 'BOA': '#66bb6a', 'REGULAR': '#ffa726',
        'RUIM': '#ff7043', 'CRITICA': '#f44336',
    }
    cores = [cores_classes.get(c, '#999') for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#f8f9fa')

    bars = ax.barh(nomes, indices, color=cores, edgecolor='white',
                   linewidth=1.5, height=0.6)

    for bar, idx, classe in zip(bars, indices, classes):
        ax.text(idx + 1, bar.get_y() + bar.get_height() / 2,
                f'{idx:.1f} — {classe}',
                va='center', fontsize=10, fontweight='bold', color='#1a1a2e')

    # Faixas de classificacao
    faixas = [(0, 20, '#f44336'), (20, 40, '#ff7043'), (40, 60, '#ffa726'),
              (60, 80, '#66bb6a'), (80, 100, '#26c6da')]
    for xmin, xmax, cor in faixas:
        ax.axvspan(xmin, xmax, alpha=0.06, color=cor)

    ax.set_xlim(0, 105)
    ax.set_xlabel('Indice de Qualidade', fontsize=11)
    ax.set_title('Resultado do Sistema Fuzzy — Todas as Amostras', fontsize=13,
                 fontweight='bold')
    ax.axvline(40, color='gray', lw=1, ls='--', alpha=0.5)
    ax.axvline(60, color='gray', lw=1, ls='--', alpha=0.5)
    ax.axvline(80, color='gray', lw=1, ls='--', alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 65)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 5: SISTEMA COMPLETO COM SCIKIT-FUZZY")
    print("=" * 65)
    print()
    print("Construindo o sistema fuzzy...")
    pH, turbidez, od, temp, qualidade, simulacao = construir_sistema()
    print("Sistema construido com 15 regras.")
    print()

    # Visualizar MFs
    print("Visualizando funcoes de pertinencia...")
    visualizar_mfs_skfuzzy(pH, turbidez, od, temp, qualidade)

    # Amostras de teste
    amostras = [
        ('Manancial A (preservado)', 7.2, 8,   9.5, 20),
        ('Rio B (zona urbana)',      6.8, 55,  6.2, 24),
        ('Lago C (eutrofizado)',     5.5, 130, 2.8, 31),
        ('Reservatorio D',          8.1, 22,  8.0, 18),
        ('Efluente E (industrial)', 4.2, 180, 1.2, 38),
    ]

    print("AVALIANDO TODAS AS AMOSTRAS:")
    print("=" * 65)
    print(f"  {'Amostra':<28} {'Indice':>8}  {'Classificacao'}")
    print("-" * 65)

    resultados = []
    for nome, ph_v, t_v, o_v, te_v in amostras:
        indice, classe = avaliar_amostra(simulacao, nome, ph_v, t_v, o_v, te_v)
        print(f"  {nome:<28} {indice:>8.1f}  {classe}")
        resultados.append({'nome': nome, 'indice': indice, 'classe': classe,
                           'ph': ph_v, 'turb': t_v, 'od': o_v, 'temp': te_v})
    print()

    visualizar_resultados(resultados)

    print("PROXIMO PASSO: Parte 6 — Analise e Visualizacao Avancada")
    print("  -> GO1130-Tutorial06-AnaliseEVisualizacao.py")
    print()
