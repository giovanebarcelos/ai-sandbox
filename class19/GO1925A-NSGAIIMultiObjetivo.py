# GO1925A-NSGAIIMultiObjetivo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np

class CarDesignProblem(Problem):
    """Otimizar design carro (multi-objetivo)"""
    def __init__(self):
        super().__init__(
            n_var=3,  # 3 variáveis: [peso, aerodinâmica, potência_motor]
            n_obj=2,  # 2 objetivos: consumo, velocidade
            n_constr=1,  # 1 restrição: peso > 800kg
            xl=np.array([800, 0.2, 50]),   # Limites inferiores
            xu=np.array([1500, 0.4, 200])  # Limites superiores
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """Avaliar objetivos e restrições"""
        peso = X[:, 0]
        coef_aero = X[:, 1]
        potencia = X[:, 2]

        # Objetivo 1: Consumo (L/100km) - MINIMIZAR
        consumo = 0.01 * peso + 50 / coef_aero - 0.05 * potencia

        # Objetivo 2: Velocidade máxima (km/h) - MAXIMIZAR
        # Mas NSGA-II minimiza, então usar NEGATIVO
        velocidade = 50 + 0.1 * potencia - 0.02 * peso + 100 * coef_aero

        # Restrições: peso ≥ 900kg (segurança)
        g1 = 900 - peso

        out["F"] = np.column_stack([consumo, -velocidade])  # Minimizar ambos
        out["G"] = np.column_stack([g1])


# Executar NSGA-II
problem = CarDesignProblem()

algorithm = NSGA2(
    pop_size=100,
    eliminate_duplicates=True
)

result = minimize(
    problem,
    algorithm,
    ('n_gen', 100),
    seed=42,
    verbose=True
)

# Extrair Pareto front
pareto_X = result.X  # Designs (variáveis)
pareto_F = result.F  # Objetivos (consumo, -velocidade)

print(f"\n🏎️ NSGA-II - Design Carro (Multi-objetivo):")
print(f"  Pareto front: {len(pareto_X)} soluções não-dominadas")
print(f"\nExemplos tradeoffs:")
for i in [0, len(pareto_X)//2, -1]:
    peso, aero, pot = pareto_X[i]
    consumo, vel_neg = pareto_F[i]
    print(f"  Solução {i+1}: Peso={peso:.0f}kg, Potência={pot:.0f}hp")
    print(f"           → Consumo={consumo:.2f} L/100km, Velocidade={-vel_neg:.0f} km/h")

# Visualizar Pareto front
plot = Scatter()
plot.add(pareto_F, color="red")
plot.xlabel("Consumo (L/100km)")
plot.ylabel("Velocidade (km/h, negativa)")
plot.title("Pareto Front - Design Carro")
plot.show()
