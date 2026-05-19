# GO1120-Recursos
# SNIPPET MÍNIMO de scikit-fuzzy — esqueleto para iniciar um sistema fuzzy.
# Mostra apenas a estrutura básica: uma entrada, uma saída, uma regra.
#
# Este código é INCOMPLETO (note os '...' em rule1 e system).
# Para um exemplo completo e funcional, veja:
#   GO1108 — sistema básico de gorjeta
#   GO1110 — sistema completo com múltiplos cenários de teste
#   GO1111/GO1112 — controlador de temperatura (projeto completo)
#
# Para outro problema: substitua 'temperatura'/'potencia' pelas variáveis
# do seu domínio e defina as MFs e regras correspondentes.
import skfuzzy as fuzz
from skfuzzy import control as ctrl


if __name__ == "__main__":
    temp = ctrl.Antecedent(np.arange(0, 41, 1), 'temperatura')
    pot = ctrl.Consequent(np.arange(0, 101, 1), 'potencia')
    temp['baixa'] = fuzz.trimf(temp.universe, [0, 0, 20])
    # ... definir demais
    rule1 = ctrl.Rule(temp['baixa'], pot['alta'])
    system = ctrl.ControlSystem([rule1, ...])
    sim = ctrl.ControlSystemSimulation(system)
