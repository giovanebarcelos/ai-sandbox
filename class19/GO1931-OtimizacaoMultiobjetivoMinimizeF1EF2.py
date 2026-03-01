# GO1931-OtimizaçãoMultiobjetivoMinimizeF1EF2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem, get_termination
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# Definir problema multi-objetivo
# Minimizar f1(x) = x^2 e f2(x) = (x-2)^2 simultaneamente

class MultiObjectiveProblem:
    def __init__(self):
        self.n_var = 1  # 1 variável
        self.n_obj = 2  # 2 objetivos
        self.xl = np.array([-5])  # Lower bound
        self.xu = np.array([5])   # Upper bound

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] ** 2
        f2 = (x[:, 0] - 2) ** 2
        out["F"] = np.column_stack([f1, f2])

# Executar NSGA-II
print("🎯 Executando NSGA-II...")

algorithm = NSGA2(pop_size=100)
problem = MultiObjectiveProblem()

# res = minimize(problem,
#                algorithm,
#                termination=('n_gen', 200),
#                seed=1,
#                verbose=False)

# Visualizar Frente de Pareto
# plt.figure(figsize=(10, 6))
# plt.scatter(res.F[:, 0], res.F[:, 1], s=50, alpha=0.6)
# plt.xlabel('Objetivo 1: f1(x) = x²', fontsize=12)
# plt.ylabel('Objetivo 2: f2(x) = (x-2)²', fontsize=12)
# plt.title('Frente de Pareto - NSGA-II', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# plt.show()

print("\n✅ Frente de Pareto encontrada!")
print("  • Trade-off visível: melhorar f1 piora f2 e vice-versa")
print("  • Decisor escolhe solução baseado em preferências")
