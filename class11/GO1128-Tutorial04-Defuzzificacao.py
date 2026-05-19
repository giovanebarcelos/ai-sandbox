# =============================================================================
# Identificador: GO1128-Tutorial04-Defuzzificacao
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 4 DE 7 — DEFUZZIFICACAO E CLASSIFICACAO FINAL
#
# A inferencia (Parte 3) produz um CONJUNTO FUZZY de saida — uma curva
# que representa a incerteza do resultado. Mas para tomar uma acao
# concreta (emitir alerta, liberar agua, etc.), precisamos de um
# NUMERO. Isso e a Defuzzificacao.
#
# Os 5 metodos mais comuns:
#
#   1. CENTROIDE (COG)  — centro de massa da area. MAIS USADO.
#   2. BISECTOR (BOA)   — divide a area ao meio
#   3. MOM              — media dos valores com pertinencia maxima
#   4. SOM              — menor valor com pertinencia maxima
#   5. LOM              — maior valor com pertinencia maxima
#
# Nesta parte voce vai:
#   1. Implementar os 5 metodos de defuzzificacao
#   2. Comparar os resultados para a mesma entrada
#   3. Converter o indice numerico em classificacao linguistica
#   4. Visualizar como cada metodo encontra seu valor
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt
try:
    get_ipython()
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    pass

import numpy as np


def trapmf(x_array, a, b, c, d):
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


def pt(x, a, b, c, d):
    return float(trapmf(np.array([x]), a, b, c, d)[0])


# MFs de entrada e saida (idem partes anteriores)
PH_ACIDO,    PH_NEUTRO,    PH_ALCALINO  = (4,4,5.5,6.5), (6,7,7.5,8.5), (7.5,9,10,10)
TURB_CLARA,  TURB_MODERADA,TURB_TURVA   = (0,0,20,50),   (30,60,80,120),(90,140,200,200)
OD_BAIXO,    OD_MEDIO,     OD_ALTO      = (0,0,3,5),      (4,6,7,9),     (8,10,14,14)
TEMP_FRIA,   TEMP_IDEAL,   TEMP_QUENTE  = (0,0,10,18),    (15,20,24,28), (25,32,40,40)
IQ_CRITICA,  IQ_RUIM = (0,0,10,20), (15,25,30,40)
IQ_REGULAR,  IQ_BOA  = (35,45,55,65), (60,70,75,82)
IQ_OTIMA             = (78,88,100,100)


def avaliar_regras(ph, turb, od, temp):
    ph_ac = pt(ph, *PH_ACIDO);    ph_ne = pt(ph, *PH_NEUTRO)
    ph_al = pt(ph, *PH_ALCALINO)
    t_cl  = pt(turb, *TURB_CLARA);t_mo = pt(turb, *TURB_MODERADA)
    t_tu  = pt(turb, *TURB_TURVA)
    od_bx = pt(od, *OD_BAIXO);    od_md = pt(od, *OD_MEDIO)
    od_al = pt(od, *OD_ALTO)
    te_fr = pt(temp, *TEMP_FRIA);  te_id = pt(temp, *TEMP_IDEAL)
    te_qt = pt(temp, *TEMP_QUENTE)
    return [
        (min(ph_ne, t_cl, od_al),        'Otima',   IQ_OTIMA),
        (min(ph_ne, t_cl, od_md, te_id), 'Otima',   IQ_OTIMA),
        (min(ph_ne, t_mo, od_al),        'Boa',     IQ_BOA),
        (min(ph_ne, t_mo, od_md),        'Boa',     IQ_BOA),
        (min(ph_al, t_cl, od_al),        'Boa',     IQ_BOA),
        (min(ph_ac, t_mo, od_md),        'Regular', IQ_REGULAR),
        (min(ph_ne, t_mo, od_bx),        'Regular', IQ_REGULAR),
        (min(ph_ne, t_tu, od_md),        'Regular', IQ_REGULAR),
        (min(ph_al, t_mo, od_md),        'Regular', IQ_REGULAR),
        (min(ph_ac, t_tu, od_md),        'Ruim',    IQ_RUIM),
        (min(ph_al, t_tu, od_md),        'Ruim',    IQ_RUIM),
        (min(ph_ne, t_tu, od_bx),        'Ruim',    IQ_RUIM),
        (min(ph_ac, t_tu, od_bx),        'Critica', IQ_CRITICA),
        (min(ph_al, t_tu, od_bx),        'Critica', IQ_CRITICA),
        (min(te_qt, od_bx),              'Critica', IQ_CRITICA),
    ]


def agregar(ph, turb, od, temp, n_pts=1000):
    """Retorna o conjunto fuzzy agregado (array mu)."""
    x = np.linspace(0, 100, n_pts)
    mu = np.zeros(n_pts)
    for forca, _, mf in avaliar_regras(ph, turb, od, temp):
        mu = np.maximum(mu, np.minimum(forca, trapmf(x, *mf)))
    return x, mu


# =============================================================================
# METODOS DE DEFUZZIFICACAO
# =============================================================================

def defuzz_centroide(x, mu):
    """Centro de massa (COG). O mais usado em sistemas reais."""
    soma_mu = np.sum(mu)
    if soma_mu < 1e-10:
        return 0.0
    return float(np.sum(x * mu) / soma_mu)


def defuzz_bisector(x, mu):
    """Divide a area em duas metades iguais."""
    area_total = np.trapz(mu, x)
    if area_total < 1e-10:
        return 0.0
    dx = x[1] - x[0]
    area_acum = 0.0
    for i, (xi, mi) in enumerate(zip(x[:-1], mu[:-1])):
        area_acum += (mi + mu[i + 1]) / 2 * dx
        if area_acum >= area_total / 2:
            return float(xi)
    return float(x[-1])


def defuzz_mom(x, mu):
    """Media dos maximos."""
    mu_max = np.max(mu)
    if mu_max < 1e-10:
        return 0.0
    indices = np.where(np.isclose(mu, mu_max, atol=1e-6))[0]
    return float(np.mean(x[indices]))


def defuzz_som(x, mu):
    """Menor dos maximos."""
    mu_max = np.max(mu)
    if mu_max < 1e-10:
        return 0.0
    indices = np.where(np.isclose(mu, mu_max, atol=1e-6))[0]
    return float(x[indices[0]])


def defuzz_lom(x, mu):
    """Maior dos maximos."""
    mu_max = np.max(mu)
    if mu_max < 1e-10:
        return 0.0
    indices = np.where(np.isclose(mu, mu_max, atol=1e-6))[0]
    return float(x[indices[-1]])


# =============================================================================
# CLASSIFICACAO DO INDICE
# =============================================================================

CLASSIFICACOES = [
    (80, 100, 'OTIMA',   '#26c6da', 'Agua potavel sem restricoes'),
    (60,  80, 'BOA',     '#66bb6a', 'Agua adequada para consumo'),
    (40,  60, 'REGULAR', '#ffa726', 'Requer tratamento basico'),
    (20,  40, 'RUIM',    '#ff7043', 'Requer tratamento avancado'),
    ( 0,  20, 'CRITICA', '#f44336', 'Imprópria — intervencao urgente'),
]


def classificar(indice):
    for minv, maxv, nome, cor, acao in CLASSIFICACOES:
        if minv <= indice <= maxv:
            return nome, cor, acao
    return 'INDEFINIDA', '#999', 'Verificar dados'


def comparar_metodos(nome_amostra, ph, turb, od, temp):
    """Compara os 5 metodos de defuzzificacao para a mesma entrada."""
    x, mu = agregar(ph, turb, od, temp)

    resultados = {
        'Centroide (COG)': defuzz_centroide(x, mu),
        'Bisector  (BOA)': defuzz_bisector(x, mu),
        'MOM':             defuzz_mom(x, mu),
        'SOM':             defuzz_som(x, mu),
        'LOM':             defuzz_lom(x, mu),
    }

    print(f"\n{'='*60}")
    print(f"  DEFUZZIFICACAO — {nome_amostra}")
    print(f"  pH={ph}  Turb={turb}  OD={od}  Temp={temp}")
    print(f"{'='*60}")
    print(f"  {'Metodo':<20} {'Indice':>8}  {'Classificacao'}")
    print("-" * 60)
    for metodo, valor in resultados.items():
        classe, _, _ = classificar(valor)
        print(f"  {metodo:<20} {valor:>8.1f}  {classe}")
    print()

    # Grafico
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#f8f9fa')
    ax.fill_between(x, mu, alpha=0.35, color='#1565c0', label='Conjunto agregado')
    ax.plot(x, mu, lw=2, color='#1565c0')

    cores_metodos = ['#e53935', '#fb8c00', '#43a047', '#9c27b0', '#00897b']
    estilos = ['-', '--', ':', '-.', (0, (5, 2))]
    for (metodo, valor), cor, ls in zip(resultados.items(), cores_metodos, estilos):
        ax.axvline(valor, color=cor, lw=2, ls=ls, label=f'{metodo}: {valor:.1f}')

    # Faixas de classificacao
    for minv, maxv, nome_c, cor, _ in CLASSIFICACOES:
        ax.axvspan(minv, maxv, alpha=0.05, color=cor)
        ax.text((minv + maxv) / 2, 0.02, nome_c, ha='center', fontsize=7,
                color=cor, rotation=90, va='bottom')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Indice de Qualidade', fontsize=11)
    ax.set_ylabel('Grau de pertinencia', fontsize=11)
    ax.set_title(f'Comparacao de Metodos de Defuzzificacao — {nome_amostra}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Classificacao usando Centroide (metodo padrao)
    indice_final = resultados['Centroide (COG)']
    classe, cor_classe, acao = classificar(indice_final)
    print(f"  RESULTADO FINAL (Centroide): {indice_final:.1f} => {classe}")
    print(f"  Acao recomendada: {acao}")
    return indice_final, classe


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 4: DEFUZZIFICACAO E CLASSIFICACAO FINAL")
    print("=" * 60)
    print()
    print("ESCALA DE CLASSIFICACAO:")
    print("-" * 55)
    for minv, maxv, nome, _, acao in CLASSIFICACOES:
        print(f"  [{minv:3d}-{maxv:3d}]  {nome:<10}: {acao}")
    print()

    amostras = [
        ('Manancial A (preservado)', 7.2, 8,   9.5, 20),
        ('Rio B (zona urbana)',      6.8, 55,  6.2, 24),
        ('Lago C (eutrofizado)',     5.5, 130, 2.8, 31),
        ('Efluente E (industrial)', 4.2, 180, 1.2, 38),
    ]

    for nome, ph, turb, od, temp in amostras:
        comparar_metodos(nome, ph, turb, od, temp)

    print()
    print("PROXIMO PASSO: Parte 5 — Sistema Completo com scikit-fuzzy")
    print("  -> GO1129-Tutorial05-SistemaCompleto.py")
    print()
