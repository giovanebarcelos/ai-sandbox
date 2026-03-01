# GO1108-SistemaDeControle
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# 1. Criar universos


if __name__ == "__main__":
    qualidade_universo = np.arange(0, 11, 1)
    servico_universo = np.arange(0, 11, 1)
    gorjeta_universo = np.arange(0, 26, 1)

    # 2. Criar variáveis fuzzy
    qualidade = ctrl.Antecedent(qualidade_universo, 'qualidade')
    servico = ctrl.Antecedent(servico_universo, 'servico')
    gorjeta = ctrl.Consequent(gorjeta_universo, 'gorjeta')

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

    # 5. Definir regras
    regra1 = ctrl.Rule(qualidade['ruim'] | servico['poor'], gorjeta['poor'])
    regra2 = ctrl.Rule(servico['average'], gorjeta['average'])
    regra3 = ctrl.Rule(servico['good'] | qualidade['boa'], gorjeta['good'])

    # 6. Criar sistema de controle
    sistema_gorjeta = ctrl.ControlSystem([regra1, regra2, regra3])

    # 7. Criar simulação
    simulacao = ctrl.ControlSystemSimulation(sistema_gorjeta)

    # 8. Fornecer entradas
    simulacao.input['qualidade'] = 6.5
    simulacao.input['servico'] = 9.8

    # 9. Computar resultado
    simulacao.compute()

    # 10. Obter saída
    print(f"Gorjeta: {simulacao.output['gorjeta']:.1f}%")

    # 11. Visualizar resultado
    gorjeta.view(sim=simulacao)
    plt.show()
