# GO1120-Recursos
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
