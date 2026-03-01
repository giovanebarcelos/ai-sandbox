# GO1110-OuSuperiorRecomendadoPython11
"""
Sistema de Gorjetas Fuzzy - Projeto Completo
Calcula porcentagem de gorjeta baseado em qualidade e serviço
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# =============================================================================
# 1. DEFINIR VARIÁVEIS E UNIVERSOS
# =============================================================================

# Variáveis de entrada
qualidade = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')

# Variável de saída
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')

# =============================================================================
# 2. FUNÇÕES DE PERTINÊNCIA
# =============================================================================

# Qualidade (triangulares)
qualidade['ruim'] = fuzz.trimf(qualidade.universe, [0, 0, 5])
qualidade['media'] = fuzz.trimf(qualidade.universe, [0, 5, 10])
qualidade['boa'] = fuzz.trimf(qualidade.universe, [5, 10, 10])

# Serviço (triangulares)
servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['medio'] = fuzz.trimf(servico.universe, [0, 5, 10])
servico['bom'] = fuzz.trimf(servico.universe, [5, 10, 10])

# Gorjeta (triangulares)
gorjeta['baixa'] = fuzz.trimf(gorjeta.universe, [0, 0, 13])
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [0, 13, 25])
gorjeta['alta'] = fuzz.trimf(gorjeta.universe, [13, 25, 25])

# =============================================================================
# 3. VISUALIZAR FUNÇÕES DE PERTINÊNCIA
# =============================================================================

qualidade.view()
plt.title('Qualidade da Comida')
plt.tight_layout()

servico.view()
plt.title('Qualidade do Serviço')
plt.tight_layout()

gorjeta.view()
plt.title('Porcentagem de Gorjeta')
plt.tight_layout()

# =============================================================================
# 4. DEFINIR BASE DE REGRAS
# =============================================================================

regra1 = ctrl.Rule(
    qualidade['ruim'] | servico['ruim'],
    gorjeta['baixa']
)
regra2 = ctrl.Rule(
    servico['medio'],
    gorjeta['media']
)
regra3 = ctrl.Rule(
    servico['bom'] | qualidade['boa'],
    gorjeta['alta']
)

# Regras adicionais para maior precisão
regra4 = ctrl.Rule(
    qualidade['media'] & servico['medio'],
    gorjeta['media']
)
regra5 = ctrl.Rule(
    qualidade['boa'] & servico['bom'],
    gorjeta['alta']
)

# =============================================================================
# 5. CRIAR E SIMULAR SISTEMA
# =============================================================================

sistema = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5])
simulacao = ctrl.ControlSystemSimulation(sistema)

# =============================================================================
# 6. TESTAR COM DIFERENTES ENTRADAS
# =============================================================================

def calcular_gorjeta(qual, serv):
    simulacao.input['qualidade'] = qual
    simulacao.input['servico'] = serv
    simulacao.compute()
    return simulacao.output['gorjeta']

# Casos de teste
casos = [
    (2, 3),    # Baixa qualidade e serviço
    (5, 5),    # Médios
    (8, 9),    # Altos
    (10, 10),  # Perfeitos
    (3, 9),    # Qualidade ruim, serviço bom
    (9, 3),    # Qualidade boa, serviço ruim
]

print("\n╔════════════════════════════════════════════════════════╗")
print("║    RESULTADOS DO SISTEMA DE GORJETAS                   ║")
print("╠════════════════════════════════════════════════════════╣")
for qual, serv in casos:
    gor = calcular_gorjeta(qual, serv)
    print(f"║  Qualidade: {qual:2d} | Serviço: {serv:2d} | Gorjeta: {gor:5.1f}%  ║")
print("╚════════════════════════════════════════════════════════╝\n")

# =============================================================================
# 7. VISUALIZAR EXEMPLO ESPECÍFICO
# =============================================================================

simulacao.input['qualidade'] = 6.5
simulacao.input['servico'] = 9.8
simulacao.compute()
print(f"Exemplo: Qualidade=6.5, Serviço=9.8")
print(f"Gorjeta recomendada: {simulacao.output['gorjeta']:.1f}%")

# Visualizar ativação das regras
gorjeta.view(sim=simulacao)
plt.title('Resultado da Inferência Fuzzy')
plt.tight_layout()
plt.show()
