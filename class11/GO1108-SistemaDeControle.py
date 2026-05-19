# GO1108-SistemaDeControle
# Demonstra o uso da biblioteca scikit-fuzzy para construir um sistema de controle fuzzy.
# Instala scikit-fuzzy automaticamente se não estiver disponível (Colab / Jupyter).
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    import sys, subprocess
    # scikit-fuzzy requer scipy (funções matemáticas) e networkx (grafo de regras)
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-fuzzy", "scipy", "networkx", "-q"], check=True)
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

if __name__ == "__main__":
    # BLOCO 1 — UNIVERSOS: definem os intervalos de cada variável.
    # Para outro problema: ajuste os arange() para o seu domínio.
    qualidade_universo = np.arange(0, 11, 1)
    servico_universo   = np.arange(0, 11, 1)
    gorjeta_universo   = np.arange(0, 26, 1)

    # BLOCO 2 — VARIÁVEIS FUZZY
    # Antecedent = entrada; Consequent = saída.
    qualidade = ctrl.Antecedent(qualidade_universo, 'qualidade')
    servico   = ctrl.Antecedent(servico_universo,   'servico')
    gorjeta   = ctrl.Consequent(gorjeta_universo,   'gorjeta')

    # BLOCO 3 — FUNÇÕES DE PERTINÊNCIA
    # automf(3) gera 3 termos triangulares uniformes: 'poor' / 'average' / 'good'.
    # Para controle fino, defina manualmente com trimf/trapmf/gaussmf (exemplo abaixo).
    servico.automf(3)
    gorjeta.automf(3)

    # Definição manual para qualidade (sobrescreve automf):
    # fuzz.trimf(universo, [a, b, c]) — triângulo com pico em b, zero em a e c
    qualidade['ruim']  = fuzz.trimf(qualidade.universe, [0, 0, 5])
    qualidade['media'] = fuzz.trimf(qualidade.universe, [0, 5, 10])
    qualidade['boa']   = fuzz.trimf(qualidade.universe, [5, 10, 10])

    # Visualizar funções de pertinência
    qualidade.view()
    servico.view()
    gorjeta.view()
    plt.show()

    # BLOCO 4 — REGRAS: conectam antecedentes (|=OR, &=AND) ao consequente.
    # Para outro problema: reescreva as regras com os termos do seu domínio.
    regra1 = ctrl.Rule(qualidade['ruim']  | servico['poor'],    gorjeta['poor'])
    regra2 = ctrl.Rule(servico['average'],                      gorjeta['average'])
    regra3 = ctrl.Rule(servico['good']    | qualidade['boa'],   gorjeta['good'])

    # BLOCO 5 — SISTEMA DE CONTROLE: une as regras em um sistema executável.
    sistema_gorjeta = ctrl.ControlSystem([regra1, regra2, regra3])
    simulacao       = ctrl.ControlSystemSimulation(sistema_gorjeta)

    # BLOCO 6 — ENTRADA DE TESTE: qualidade=6.5, serviço=9.8 (exemplo do slide).
    # Para outro problema: substitua os nomes das variáveis e os valores.
    simulacao.input['qualidade'] = 6.5
    simulacao.input['servico']   = 9.8

    # compute() executa fuzzificação → inferência → agregação → defuzzificação
    simulacao.compute()

    print(f"Gorjeta: {simulacao.output['gorjeta']:.1f}%")

    # Visualiza a saída com o centroide marcado
    gorjeta.view(sim=simulacao)
    plt.show()
