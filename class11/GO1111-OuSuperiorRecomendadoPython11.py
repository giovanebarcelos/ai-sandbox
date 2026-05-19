# GO1111-OuSuperiorRecomendadoPython11

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    """
    Controlador de Temperatura Fuzzy - Parte 1: Setup e Funções de Pertinência
    """
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

    # =============================================================================
    # 1. CRIAR VARIÁVEIS FUZZY
    # =============================================================================

    # BLOCO 1 — VARIÁVEIS: temperatura [0,50°C], umidade [0,100%], potência [-100,+100].
    # Para outro problema: substitua nomes e intervalos. O sinal da potência
    # (negativo=aquece, positivo=resfria) é uma convenção deste controlador.
    # Entradas
    temperatura = ctrl.Antecedent(np.arange(0, 51, 1), 'temperatura')
    umidade = ctrl.Antecedent(np.arange(0, 101, 1), 'umidade')

    # Saída (negativo=aquece, positivo=resfria)
    potencia = ctrl.Consequent(np.arange(-100, 101, 1), 'potencia')

    # BLOCO 2 — MFs de TEMPERATURA: 5 termos cobrem toda a faixa.
    # Para outro problema: ajuste o número de termos e os parâmetros trapmf/trimf.
    # Mais termos = controle mais fino, porém mais regras na parte 2 (GO1112).
    # =============================================================================
    # 2. FUNÇÕES DE PERTINÊNCIA - TEMPERATURA
    # =============================================================================

    temperatura['muito_fria'] = fuzz.trapmf(temperatura.universe, [0, 0, 10, 15])
    temperatura['fria'] = fuzz.trimf(temperatura.universe, [12, 16, 20])
    temperatura['agradavel'] = fuzz.trimf(temperatura.universe, [18, 22, 26])
    temperatura['quente'] = fuzz.trimf(temperatura.universe, [24, 28, 32])
    temperatura['muito_quente'] = fuzz.trapmf(temperatura.universe, [30, 35, 50, 50])

    # =============================================================================
    # 3. FUNÇÕES DE PERTINÊNCIA - UMIDADE
    # =============================================================================

    umidade['seca'] = fuzz.trapmf(umidade.universe, [0, 0, 30, 40])
    umidade['normal'] = fuzz.trimf(umidade.universe, [30, 50, 70])
    umidade['umida'] = fuzz.trapmf(umidade.universe, [60, 70, 100, 100])

    # BLOCO 3 — MFs de POTÊNCIA (SAÍDA): intervalo simétrico [-100, +100].
    # Para outro problema: defina os termos da sua saída (ex: velocidade do motor,
    # dose de medicamento, ângulo de deflexão…).
    # =============================================================================
    # 4. FUNÇÕES DE PERTINÊNCIA - POTÊNCIA
    # =============================================================================

    potencia['aquecer_forte'] = fuzz.trapmf(potencia.universe, [-100, -100, -75, -50])
    potencia['aquecer_leve'] = fuzz.trimf(potencia.universe, [-60, -30, 0])
    potencia['desligado'] = fuzz.trimf(potencia.universe, [-20, 0, 20])
    potencia['resfriar_leve'] = fuzz.trimf(potencia.universe, [0, 30, 60])
    potencia['resfriar_forte'] = fuzz.trapmf(potencia.universe, [50, 75, 100, 100])

    # =============================================================================
    # 5. VISUALIZAR FUNÇÕES DE PERTINÊNCIA
    # =============================================================================

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    temperatura.view(ax=axes[0])
    axes[0].set_title('Funções de Pertinência - Temperatura')

    umidade.view(ax=axes[1])
    axes[1].set_title('Funções de Pertinência - Umidade')

    potencia.view(ax=axes[2])
    axes[2].set_title('Funções de Pertinência - Potência')

    plt.tight_layout()
    plt.show()

    # [Continua no próximo slide com as regras...]
