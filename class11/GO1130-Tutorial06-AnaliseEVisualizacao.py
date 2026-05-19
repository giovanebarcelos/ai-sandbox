# =============================================================================
# Identificador: GO1130-Tutorial06-AnaliseEVisualizacao
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 6 DE 7 — ANALISE E VISUALIZACAO AVANCADA
#
# Com o sistema funcionando, esta parte explora visualizacoes avancadas
# que revelam o comportamento do sistema fuzzy:
#
#   1. Superficie de controle (como a saida varia com 2 entradas)
#   2. Analise de sensibilidade (qual variavel mais influencia o resultado)
#   3. Monitoramento temporal (simulacao de 24 horas de coleta)
#   4. Mapa de calor do indice de qualidade vs pH e Turbidez
#
# Essas visualizacoes sao essenciais em sistemas reais para:
#   - Validar que o sistema se comporta como esperado
#   - Identificar regioes de fronteira criticas
#   - Comunicar resultados para nao-especialistas
#
# Nesta parte voce vai:
#   1. Gerar a superficie 3D pH x Turbidez -> Qualidade
#   2. Fazer analise de sensibilidade variando cada parametro
#   3. Simular monitoramento temporal com ruido realista
#   4. Gerar mapa de calor interativo
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    print("Instale: pip install scikit-fuzzy")
    raise


# =============================================================================
# CONSTRUCAO DO SISTEMA (idem Parte 5)
# =============================================================================

def construir_sistema():
    ph_u    = np.arange(4.0, 10.1, 0.1)
    turb_u  = np.arange(0.0, 201.0, 2.0)
    od_u    = np.arange(0.0, 14.1, 0.2)
    temp_u  = np.arange(0.0, 40.1, 0.5)
    qual_u  = np.arange(0.0, 100.1, 0.5)

    pH       = ctrl.Antecedent(ph_u,   'pH')
    turbidez = ctrl.Antecedent(turb_u, 'Turbidez')
    od_v     = ctrl.Antecedent(od_u,   'OD')
    temp     = ctrl.Antecedent(temp_u, 'Temperatura')
    qualidade= ctrl.Consequent(qual_u, 'Qualidade', defuzzify_method='centroid')

    pH['Acido']    = fuzz.trapmf(ph_u,   [4.0, 4.0, 5.5, 6.5])
    pH['Neutro']   = fuzz.trapmf(ph_u,   [6.0, 7.0, 7.5, 8.5])
    pH['Alcalino'] = fuzz.trapmf(ph_u,   [7.5, 9.0,10.0,10.0])

    turbidez['Clara']    = fuzz.trapmf(turb_u, [  0,  0, 20,  50])
    turbidez['Moderada'] = fuzz.trapmf(turb_u, [ 30, 60, 80, 120])
    turbidez['Turva']    = fuzz.trapmf(turb_u, [ 90,140,200, 200])

    od_v['Baixo'] = fuzz.trapmf(od_u, [0, 0,  3,  5])
    od_v['Medio'] = fuzz.trapmf(od_u, [4, 6,  7,  9])
    od_v['Alto']  = fuzz.trapmf(od_u, [8,10, 14, 14])

    temp['Fria']   = fuzz.trapmf(temp_u, [ 0,  0, 10, 18])
    temp['Ideal']  = fuzz.trapmf(temp_u, [15, 20, 24, 28])
    temp['Quente'] = fuzz.trapmf(temp_u, [25, 32, 40, 40])

    qualidade['Critica'] = fuzz.trapmf(qual_u, [ 0,  0, 10, 20])
    qualidade['Ruim']    = fuzz.trapmf(qual_u, [15, 25, 30, 40])
    qualidade['Regular'] = fuzz.trapmf(qual_u, [35, 45, 55, 65])
    qualidade['Boa']     = fuzz.trapmf(qual_u, [60, 70, 75, 82])
    qualidade['Otima']   = fuzz.trapmf(qual_u, [78, 88,100,100])

    regras = [
        ctrl.Rule(pH['Neutro']   & turbidez['Clara']    & od_v['Alto'],                  qualidade['Otima']),
        ctrl.Rule(pH['Neutro']   & turbidez['Clara']    & od_v['Medio'] & temp['Ideal'], qualidade['Otima']),
        ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od_v['Alto'],                  qualidade['Boa']),
        ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od_v['Medio'],                 qualidade['Boa']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Clara']    & od_v['Alto'],                  qualidade['Boa']),
        ctrl.Rule(pH['Acido']    & turbidez['Moderada'] & od_v['Medio'],                 qualidade['Regular']),
        ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od_v['Baixo'],                 qualidade['Regular']),
        ctrl.Rule(pH['Neutro']   & turbidez['Turva']    & od_v['Medio'],                 qualidade['Regular']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Moderada'] & od_v['Medio'],                 qualidade['Regular']),
        ctrl.Rule(pH['Acido']    & turbidez['Turva']    & od_v['Medio'],                 qualidade['Ruim']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Turva']    & od_v['Medio'],                 qualidade['Ruim']),
        ctrl.Rule(pH['Neutro']   & turbidez['Turva']    & od_v['Baixo'],                 qualidade['Ruim']),
        ctrl.Rule(pH['Acido']    & turbidez['Turva']    & od_v['Baixo'],                 qualidade['Critica']),
        ctrl.Rule(pH['Alcalino'] & turbidez['Turva']    & od_v['Baixo'],                 qualidade['Critica']),
        ctrl.Rule(temp['Quente'] & od_v['Baixo'],                                        qualidade['Critica']),
    ]

    sistema_ctrl = ctrl.ControlSystem(regras)
    return ctrl.ControlSystemSimulation(sistema_ctrl)


def avaliar_seguro(sim, ph, turb, od, temp_v):
    """Avalia o sistema capturando excecoes (valores fora de range)."""
    try:
        sim.input['pH']          = float(np.clip(ph, 4.0, 10.0))
        sim.input['Turbidez']    = float(np.clip(turb, 0, 200))
        sim.input['OD']          = float(np.clip(od, 0, 14))
        sim.input['Temperatura'] = float(np.clip(temp_v, 0, 40))
        sim.compute()
        return float(sim.output['Qualidade'])
    except Exception:
        return float('nan')


def classificar(indice):
    if indice >= 80: return 'OTIMA'
    elif indice >= 60: return 'BOA'
    elif indice >= 40: return 'REGULAR'
    elif indice >= 20: return 'RUIM'
    else: return 'CRITICA'


# =============================================================================
# 1. SUPERFICIE DE CONTROLE (pH x Turbidez -> Qualidade)
# =============================================================================

def superficie_controle(sim):
    """Gera superficie 3D: pH x Turbidez -> Indice de Qualidade."""
    print("Calculando superficie de controle (pH x Turbidez)...")
    print("  OD = 9.0 mg/L (fixo) | Temperatura = 22 C (fixa)")
    print("  Aguarde — pode demorar alguns segundos...")

    ph_vals   = np.arange(4.5, 9.6, 0.4)
    turb_vals = np.arange(5, 195, 10)
    Z = np.zeros((len(ph_vals), len(turb_vals)))

    for i, ph in enumerate(ph_vals):
        for j, turb in enumerate(turb_vals):
            Z[i, j] = avaliar_seguro(sim, ph, turb, 9.0, 22.0)

    PH_grid, TURB_grid = np.meshgrid(ph_vals, turb_vals, indexing='ij')

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(PH_grid, TURB_grid, Z, cmap='RdYlGn',
                           edgecolor='none', alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Indice de Qualidade')

    ax.set_xlabel('pH', fontsize=11, labelpad=8)
    ax.set_ylabel('Turbidez (NTU)', fontsize=11, labelpad=8)
    ax.set_zlabel('Qualidade', fontsize=11, labelpad=8)
    ax.set_title('Superficie de Controle — pH x Turbidez => Qualidade\n'
                 '(OD=9.0, Temp=22C fixos)', fontsize=12, fontweight='bold')
    ax.set_zlim(0, 100)
    plt.tight_layout()
    plt.show()
    print("  Superficie gerada.")


# =============================================================================
# 2. ANALISE DE SENSIBILIDADE
# =============================================================================

def analise_sensibilidade(sim):
    """Varia cada variavel mantendo as outras fixas — mostra impacto no indice."""
    base = {'ph': 7.0, 'turb': 40.0, 'od': 8.0, 'temp': 22.0}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('Analise de Sensibilidade — Impacto de Cada Variavel',
                 fontsize=13, fontweight='bold')

    variacoes = [
        (axes[0, 0], 'pH', np.arange(4.1, 9.9, 0.1),
         'ph', '#1e88e5', 'pH'),
        (axes[0, 1], 'Turbidez (NTU)', np.arange(2, 198, 2),
         'turb', '#fb8c00', 'Turbidez (NTU)'),
        (axes[1, 0], 'OD (mg/L)', np.arange(0.2, 13.8, 0.2),
         'od', '#43a047', 'OD (mg/L)'),
        (axes[1, 1], 'Temperatura (C)', np.arange(1, 39, 0.5),
         'temp', '#e53935', 'Temperatura (C)'),
    ]

    for ax, titulo, valores, chave, cor, xlabel in variacoes:
        resultados = []
        for v in valores:
            params = {**base, chave: v}
            idx = avaliar_seguro(sim, params['ph'], params['turb'],
                                 params['od'], params['temp'])
            resultados.append(idx)

        resultados = np.array(resultados)

        # Colorir por classificacao
        cores_pts = []
        for idx in resultados:
            c = classificar(idx)
            cor_map = {'OTIMA': '#26c6da', 'BOA': '#66bb6a', 'REGULAR': '#ffa726',
                       'RUIM': '#ff7043', 'CRITICA': '#f44336'}
            cores_pts.append(cor_map.get(c, '#999'))

        ax.set_facecolor('#f8f9fa')
        ax.plot(valores, resultados, lw=2.5, color=cor)
        ax.scatter(valores, resultados, c=cores_pts, s=20, zorder=4)

        # Linhas de fronteira
        for limite, label in [(20, 'Critica/Ruim'), (40, 'Ruim/Regular'),
                               (60, 'Regular/Boa'), (80, 'Boa/Otima')]:
            ax.axhline(limite, color='gray', lw=1, ls='--', alpha=0.5)
            ax.text(valores[-1], limite + 1, label, fontsize=7, color='gray',
                    ha='right')

        # Linha da base
        base_val = base[chave]
        base_idx = avaliar_seguro(sim, base['ph'], base['turb'], base['od'], base['temp'])
        ax.axvline(base_val, color='navy', lw=1.5, ls=':', alpha=0.6,
                   label=f'Base: {base_val}')
        ax.plot([base_val], [base_idx], 'D', color='navy', markersize=8, zorder=6)

        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('Indice de Qualidade', fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. MONITORAMENTO TEMPORAL SIMULADO (24 HORAS)
# =============================================================================

def simular_monitoramento_temporal(sim):
    """Simula 24 horas de coleta de dados com variacao realista."""
    np.random.seed(42)
    horas = np.arange(0, 24.5, 0.5)
    n = len(horas)

    # Cenario: manha boa, tarde deteriorando, noite critica, madrugada recuperando
    ph_base   = 7.0 + 0.8 * np.sin(np.pi * horas / 12) + np.random.normal(0, 0.15, n)
    turb_base = 20 + 60 * np.clip(np.sin(np.pi * (horas - 8) / 10), 0, 1) + \
                np.random.normal(0, 5, n)
    od_base   = 9.0 - 4.0 * np.clip(np.sin(np.pi * (horas - 6) / 10), 0, 1) + \
                np.random.normal(0, 0.3, n)
    temp_base = 18 + 8 * np.sin(np.pi * horas / 12) + np.random.normal(0, 0.5, n)

    # Clip para os universos
    ph_base   = np.clip(ph_base, 4.1, 9.9)
    turb_base = np.clip(turb_base, 1, 199)
    od_base   = np.clip(od_base, 0.1, 13.9)
    temp_base = np.clip(temp_base, 1, 39)

    indices = []
    print("Simulando 24 horas de monitoramento...")
    for ph, turb, od, temp_v in zip(ph_base, turb_base, od_base, temp_base):
        idx = avaliar_seguro(sim, ph, turb, od, temp_v)
        indices.append(idx)
    indices = np.array(indices)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Monitoramento Temporal — 24 Horas (Simulado)',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Variaveis de entrada
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    ax1_b = ax1.twinx()
    ax1.plot(horas, ph_base,   color='#1e88e5', lw=1.8, label='pH (esc. esq.)')
    ax1.plot(horas, od_base,   color='#43a047', lw=1.8, ls='--', label='OD mg/L (esc. esq.)')
    ax1_b.plot(horas, turb_base, color='#fb8c00', lw=1.8, label='Turbidez NTU (esc. dir.)')
    ax1_b.plot(horas, temp_base, color='#e53935', lw=1.8, ls=':', label='Temp C (esc. dir.)')
    ax1.set_ylabel('pH | OD (mg/L)', fontsize=10)
    ax1_b.set_ylabel('Turbidez (NTU) | Temp (C)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1_b.legend(loc='upper right', fontsize=8)
    ax1.set_title('Variaveis de Entrada', fontsize=11)
    ax1.grid(True, alpha=0.25)

    # Subplot 2: Indice de qualidade ao longo do tempo
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    faixas = [(0, 20, '#f44336', 0.12), (20, 40, '#ff7043', 0.08),
              (40, 60, '#ffa726', 0.06), (60, 80, '#66bb6a', 0.06),
              (80, 100, '#26c6da', 0.08)]
    for ymin, ymax, cor, alpha in faixas:
        ax2.axhspan(ymin, ymax, alpha=alpha, color=cor)

    ax2.plot(horas, indices, lw=2.5, color='#1a237e', zorder=5, label='Indice Fuzzy')
    ax2.fill_between(horas, indices, alpha=0.25, color='#1a237e')

    # Alertas (indice < 40)
    criticos = horas[indices < 40]
    if len(criticos):
        ax2.axhspan(0, 40, xmin=criticos[0] / 24, xmax=criticos[-1] / 24,
                    alpha=0.15, color='red', label='Zona de Alerta')

    ax2.set_ylabel('Indice de Qualidade', fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.set_title('Indice de Qualidade ao Longo do Tempo', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, axis='y')
    for lim in [20, 40, 60, 80]:
        ax2.axhline(lim, color='gray', lw=0.8, ls='--', alpha=0.5)

    # Subplot 3: Classificacao por hora
    ax3 = axes[2]
    ax3.set_facecolor('#f8f9fa')
    cores_classe = {'OTIMA': '#26c6da', 'BOA': '#66bb6a', 'REGULAR': '#ffa726',
                    'RUIM': '#ff7043', 'CRITICA': '#f44336'}
    for i in range(len(horas) - 1):
        classe = classificar(indices[i])
        ax3.barh(0, 0.5, left=horas[i], height=0.8,
                 color=cores_classe.get(classe, '#999'), edgecolor='none')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n)
                       for n, c in cores_classe.items()]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9,
               ncol=5, framealpha=0.9)
    ax3.set_yticks([])
    ax3.set_xlabel('Hora do Dia', fontsize=10)
    ax3.set_title('Classificacao por Periodo', fontsize=11)
    ax3.set_xlim(0, 24)

    for ax in axes:
        ax.set_xticks(range(0, 25, 2))

    plt.tight_layout()
    plt.show()

    print(f"\n  Resumo das 24 horas:")
    print(f"  Indice medio: {np.nanmean(indices):.1f}")
    print(f"  Indice minimo: {np.nanmin(indices):.1f}  ({horas[np.nanargmin(indices)]:.0f}h)")
    print(f"  Indice maximo: {np.nanmax(indices):.1f}  ({horas[np.nanargmax(indices)]:.0f}h)")
    horas_criticas = np.sum(indices < 40)
    print(f"  Horas em estado RUIM/CRITICO: {horas_criticas * 0.5:.1f}h")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 65)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 6: ANALISE E VISUALIZACAO AVANCADA")
    print("=" * 65)
    print()

    print("Construindo sistema fuzzy...")
    sim = construir_sistema()
    print("Sistema pronto.")
    print()

    superficie_controle(sim)
    analise_sensibilidade(sim)
    simular_monitoramento_temporal(sim)

    print()
    print("PROXIMO PASSO: Parte 7 — Solucao Completa (pipeline final)")
    print("  -> GO1131-Tutorial07-SolucaoCompleta.py")
    print()
