# GO1108-SistemaDeControle
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Criar universos

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    # BLOCO 1 — UNIVERSOS: definem os intervalos de cada variável.
    # Para outro problema: ajuste os arange() para o seu domínio.
    qualidade_universo = np.arange(0, 11, 1)
    servico_universo = np.arange(0, 11, 1)
    gorjeta_universo = np.arange(0, 26, 1)

    # 2. Criar variáveis fuzzy
    qualidade = ctrl.Antecedent(qualidade_universo, 'qualidade')
    servico = ctrl.Antecedent(servico_universo, 'servico')
    gorjeta = ctrl.Consequent(gorjeta_universo, 'gorjeta')

    # BLOCO 2 — MFs: automf(3) gera 3 termos automaticamente (ruim/médio/bom).
    # Para controle mais fino, use trimf/trapmf manualmente (como na linha seguinte).
    # 3. Definir funções de pertinência (auto-membership)
    qualidade.automf(3)  # Cria 3 termos automaticamente
    servico.automf(3)
    gorjeta.automf(3)

    # Ou definir manualmente:
    qualidade['ruim'] = fuzz.trimf(qualidade.universe, [0, 0, 5])
    qualidade['media'] = fuzz.trimf(qualidade.universe, [0, 5, 10])
    qualidade['boa'] = fuzz.trimf(qualidade.universe, [5, 10, 10])

    # 4. Visualizar funções de pertinência
    qualidade.view()
    servico.view()
    gorjeta.view()
    plt.show()

    # BLOCO 3 — REGRAS: conectam antecedentes (|=OR, &=AND) ao consequente.
    # Para outro problema: reescreva as regras com os termos do seu domínio.
    # 5. Definir regras
    regra1 = ctrl.Rule(qualidade['ruim'] | servico['poor'], gorjeta['poor'])
    regra2 = ctrl.Rule(servico['average'], gorjeta['average'])
    regra3 = ctrl.Rule(servico['good'] | qualidade['boa'], gorjeta['good'])

    # 6. Criar sistema de controle
    sistema_gorjeta = ctrl.ControlSystem([regra1, regra2, regra3])

    # 7. Criar simulação
    simulacao = ctrl.ControlSystemSimulation(sistema_gorjeta)

    # 8. Fornecer entradas
    # BLOCO 4 — ENTRADA DE TESTE: qualidade=6.5, serviço=9.8 (exemplo do slide).
    # Para outro problema, substitua os nomes das variáveis e os valores.
    simulacao.input['qualidade'] = 6.5
    simulacao.input['servico'] = 9.8

    # 9. Computar resultado
    simulacao.compute()

    # 10. Obter saída
    print(f"Gorjeta: {simulacao.output['gorjeta']:.1f}%")

    # 11. Visualizar resultado
    gorjeta.view(sim=simulacao)
    plt.show()
