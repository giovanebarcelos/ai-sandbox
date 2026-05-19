# =============================================================================
# Identificador: GO1127-Tutorial03-RegrasEInferencia
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 3 DE 7 — REGRAS FUZZY E PROCESSO DE INFERENCIA
#
# Com as funcoes de pertinencia prontas (Parte 2), agora precisamos
# definir as REGRAS que codificam o conhecimento do especialista.
#
# Estrutura de uma regra:
#   SE <condicao1> E <condicao2> E ... ENTAO <conclusao>
#
# Exemplo:
#   SE pH e Neutro E Turbidez e Clara E OD e Alto
#   ENTAO Qualidade e OTIMA
#
# O processo de INFERENCIA avalia todas as regras para uma entrada
# e combina os resultados — formando o conjunto fuzzy de saida.
#
# Nesta parte voce vai:
#   1. Entender a estrutura das regras fuzzy
#   2. Ver como calcular a "forca" de cada regra (T-norma)
#   3. Acompanhar a inferencia passo a passo para uma amostra real
#   4. Visualizar os conjuntos de saida de cada regra ativada
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt
try:
    get_ipython()
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    pass

import numpy as np


# Reutilizamos as MFs da Parte 2
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


# Parametros das MFs (idem Parte 2)
PH_ACIDO,    PH_NEUTRO,    PH_ALCALINO  = (4,4,5.5,6.5), (6,7,7.5,8.5), (7.5,9,10,10)
TURB_CLARA,  TURB_MODERADA,TURB_TURVA   = (0,0,20,50),   (30,60,80,120),(90,140,200,200)
OD_BAIXO,    OD_MEDIO,     OD_ALTO      = (0,0,3,5),      (4,6,7,9),     (8,10,14,14)
TEMP_FRIA,   TEMP_IDEAL,   TEMP_QUENTE  = (0,0,10,18),    (15,20,24,28), (25,32,40,40)
IQ_CRITICA,  IQ_RUIM = (0,0,10,20), (15,25,30,40)
IQ_REGULAR,  IQ_BOA  = (35,45,55,65), (60,70,75,82)
IQ_OTIMA             = (78,88,100,100)


# =============================================================================
# BASE DE REGRAS
# =============================================================================
# Cada regra e uma funcao que recebe os graus de pertinencia e retorna
# (forca_da_regra, nome_do_conjunto_de_saida, parametros_da_mf_saida)

def avaliar_regras(ph, turb, od, temp):
    """
    Avalia todas as regras e retorna lista de (forca, nome_saida, mf_saida).

    Os graus de pertinencia sao calculados internamente a partir dos valores
    crisp (numericos) das 4 variaveis de entrada.
    """
    # Fuzzificacao das entradas
    ph_ac  = pt(ph,   *PH_ACIDO)
    ph_ne  = pt(ph,   *PH_NEUTRO)
    ph_al  = pt(ph,   *PH_ALCALINO)

    t_cl   = pt(turb, *TURB_CLARA)
    t_mo   = pt(turb, *TURB_MODERADA)
    t_tu   = pt(turb, *TURB_TURVA)

    od_bx  = pt(od,   *OD_BAIXO)
    od_md  = pt(od,   *OD_MEDIO)
    od_al  = pt(od,   *OD_ALTO)

    te_fr  = pt(temp, *TEMP_FRIA)
    te_id  = pt(temp, *TEMP_IDEAL)
    te_qt  = pt(temp, *TEMP_QUENTE)

    graus = {
        'ph_acido': ph_ac, 'ph_neutro': ph_ne, 'ph_alcalino': ph_al,
        'turb_clara': t_cl, 'turb_moderada': t_mo, 'turb_turva': t_tu,
        'od_baixo': od_bx, 'od_medio': od_md, 'od_alto': od_al,
        'temp_fria': te_fr, 'temp_ideal': te_id, 'temp_quente': te_qt,
    }

    # Regras — T-norma = min (Zadeh)
    # Formato: (forca, 'nome', mf_params)
    regras = [
        # OTIMA
        (min(ph_ne, t_cl, od_al),             'Otima',   IQ_OTIMA),
        (min(ph_ne, t_cl, od_md, te_id),      'Otima',   IQ_OTIMA),
        # BOA
        (min(ph_ne, t_mo, od_al),             'Boa',     IQ_BOA),
        (min(ph_ne, t_mo, od_md),             'Boa',     IQ_BOA),
        (min(ph_al, t_cl, od_al),             'Boa',     IQ_BOA),
        # REGULAR
        (min(ph_ac, t_mo, od_md),             'Regular', IQ_REGULAR),
        (min(ph_ne, t_mo, od_bx),             'Regular', IQ_REGULAR),
        (min(ph_ne, t_tu, od_md),             'Regular', IQ_REGULAR),
        (min(ph_al, t_mo, od_md),             'Regular', IQ_REGULAR),
        # RUIM
        (min(ph_ac, t_tu, od_md),             'Ruim',    IQ_RUIM),
        (min(ph_al, t_tu, od_md),             'Ruim',    IQ_RUIM),
        (min(ph_ne, t_tu, od_bx),             'Ruim',    IQ_RUIM),
        # CRITICA
        (min(ph_ac, t_tu, od_bx),             'Critica', IQ_CRITICA),
        (min(ph_al, t_tu, od_bx),             'Critica', IQ_CRITICA),
        (min(te_qt, od_bx),                   'Critica', IQ_CRITICA),
    ]

    return regras, graus


# =============================================================================
# INFERENCIA PASSO A PASSO
# =============================================================================

def inferencia_detalhada(nome, ph, turb, od, temp):
    """Mostra o processo de inferencia de Mamdani passo a passo."""
    print()
    print("=" * 65)
    print(f"  INFERENCIA MAMDANI — {nome}")
    print("=" * 65)
    print(f"  Entradas: pH={ph}  Turb={turb} NTU  OD={od} mg/L  Temp={temp} C")
    print()

    # --- Etapa 1: Fuzzificacao ---
    print("ETAPA 1: FUZZIFICACAO")
    print("-" * 65)
    _, graus = avaliar_regras(ph, turb, od, temp)
    grupos = [
        ('pH', ['ph_acido', 'ph_neutro', 'ph_alcalino']),
        ('Turbidez', ['turb_clara', 'turb_moderada', 'turb_turva']),
        ('OD', ['od_baixo', 'od_medio', 'od_alto']),
        ('Temperatura', ['temp_fria', 'temp_ideal', 'temp_quente']),
    ]
    for varname, chaves in grupos:
        vals = [(k.split('_', 1)[1].capitalize(), graus[k]) for k in chaves]
        linha = '  '.join(f'{n}={v:.3f}' for n, v in vals)
        print(f"  {varname:<12}: {linha}")
    print()

    # --- Etapa 2: Avaliacao de Regras ---
    print("ETAPA 2: AVALIACAO DAS REGRAS (T-norma = min)")
    print("-" * 65)
    regras, _ = avaliar_regras(ph, turb, od, temp)
    regras_ativas = []
    for i, (forca, nome_saida, mf) in enumerate(regras, 1):
        if forca > 0.001:
            barra = '#' * int(forca * 20)
            print(f"  R{i:02d}: forca={forca:.3f}  |{barra}  =>  {nome_saida}")
            regras_ativas.append((forca, nome_saida, mf))

    total_ativas = len(regras_ativas)
    print(f"\n  {total_ativas} regras ativadas de {len(regras)} no total")
    print()

    # --- Etapa 3: Agregacao (union maxima) ---
    print("ETAPA 3: AGREGACAO (T-conorma = max)")
    print("-" * 65)
    x_saida = np.linspace(0, 100, 1000)
    mu_agregada = np.zeros(len(x_saida))
    for forca, nome_saida, mf in regras_ativas:
        # Corte pelo minimo (Mamdani): clipagem do conjunto de saida
        mu_regra = np.minimum(forca, trapmf(x_saida, *mf))
        # Uniao (max) de todos os conjuntos cortados
        mu_agregada = np.maximum(mu_agregada, mu_regra)

    area = np.trapz(mu_agregada, x_saida)
    print(f"  Conjunto fuzzy agregado — area total: {area:.2f}")
    print()

    return regras_ativas, x_saida, mu_agregada


def visualizar_inferencia(nome, ph, turb, od, temp):
    """Grafico completo do processo de inferencia."""
    regras_ativas, x_saida, mu_agregada = inferencia_detalhada(nome, ph, turb, od, temp)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Inferencia Mamdani — {nome}\n'
                 f'pH={ph}  Turb={turb}  OD={od}  Temp={temp}',
                 fontsize=13, fontweight='bold')

    # --- Subplot 1: Conjuntos de saida de cada regra ---
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    cores_saida = {
        'Critica': '#f44336', 'Ruim': '#ff7043',
        'Regular': '#ffa726', 'Boa': '#66bb6a', 'Otima': '#26c6da',
    }
    mfs_saida = {
        'Critica': IQ_CRITICA, 'Ruim': IQ_RUIM, 'Regular': IQ_REGULAR,
        'Boa': IQ_BOA, 'Otima': IQ_OTIMA,
    }

    # MFs originais (sem corte)
    for nome_s, mf in mfs_saida.items():
        mu = trapmf(x_saida, *mf)
        ax1.plot(x_saida, mu, lw=1.5, color=cores_saida[nome_s],
                 ls='--', alpha=0.5, label=nome_s)

    # MFs cortadas (clipadas pela forca da regra)
    for forca, nome_s, mf in regras_ativas:
        mu_clip = np.minimum(forca, trapmf(x_saida, *mf))
        ax1.fill_between(x_saida, mu_clip, alpha=0.35, color=cores_saida[nome_s])
        ax1.plot(x_saida, mu_clip, lw=2, color=cores_saida[nome_s])

    ax1.set_title('Conjuntos de Saida Clipados por Cada Regra Ativa', fontsize=11)
    ax1.set_ylabel('Pertinencia')
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend(fontsize=9, ncol=5, loc='upper right')
    ax1.grid(True, alpha=0.25)

    # --- Subplot 2: Conjunto agregado ---
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    ax2.fill_between(x_saida, mu_agregada, alpha=0.4, color='#1565c0',
                     label='Conjunto agregado')
    ax2.plot(x_saida, mu_agregada, lw=2, color='#1565c0')

    # Centroide (defuzzificacao — preview)
    if np.sum(mu_agregada) > 0:
        centroide = np.sum(x_saida * mu_agregada) / np.sum(mu_agregada)
        ax2.axvline(centroide, color='#e53935', lw=2.5, ls='--',
                    label=f'Centroide = {centroide:.1f}')
        ax2.text(centroide + 1, 0.6, f'{centroide:.1f}', color='#e53935',
                 fontsize=11, fontweight='bold')

    ax2.set_title('Conjunto Fuzzy Agregado (Uniao de Todos os Clips)', fontsize=11)
    ax2.set_xlabel('Indice de Qualidade')
    ax2.set_ylabel('Pertinencia')
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlim(0, 100)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 65)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 3: REGRAS FUZZY E INFERENCIA")
    print("=" * 65)
    print()
    print("BASE DE REGRAS DO SISTEMA: 15 regras organizadas em 5 classes")
    print()
    print("  OTIMA:   R1-R2  — pH neutro, agua clara, OD alto")
    print("  BOA:     R3-R5  — condicoes razoavelmente boas")
    print("  REGULAR: R6-R9  — algum parametro fora do ideal")
    print("  RUIM:    R10-R12 — dois ou mais parametros ruins")
    print("  CRITICA: R13-R15 — condicoes muito ruins ou perigosas")
    print()

    # Amostra 1: Manancial A (agua boa)
    visualizar_inferencia(
        'Manancial A (preservado)',
        ph=7.2, turb=8, od=9.5, temp=20
    )

    # Amostra 2: Rio B (agua regular/ruim)
    visualizar_inferencia(
        'Rio B (zona urbana)',
        ph=6.8, turb=55, od=6.2, temp=24
    )

    print("PROXIMO PASSO: Parte 4 — Defuzzificacao e Resultado Final")
    print("  -> GO1128-Tutorial04-Defuzzificacao.py")
    print()
