# GO1923-CMAESBiblioteca
import cma
import numpy as np

# Função Rosenbrock (vale estreito, difícil para GA/PSO)
def rosenbrock(x):
    """f(x) = Σ[100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    Mínimo global: f(1, 1, ..., 1) = 0
    """
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
               for i in range(len(x) - 1))

# Otimizar 10 dimensões com CMA-ES
x0 = np.random.randn(10) * 5  # Ponto inicial aleatório
sigma0 = 2.0  # Passo inicial

es = cma.CMAEvolutionStrategy(x0, sigma0, {
    'popsize': 50,
    'maxiter': 200,
    'verb_disp': 20,  # Print a cada 20 gerações
    'tolx': 1e-8
})

# Loop otimização
while not es.stop():
    solutions = es.ask()  # Gerar candidatos
    fitness_list = [rosenbrock(x) for x in solutions]
    es.tell(solutions, fitness_list)  # Atualizar distribuição
    es.disp()  # Mostrar progresso

# Melhor solução
best_solution = es.result.xbest
best_fitness = es.result.fbest

print(f"\n🎯 CMA-ES - Rosenbrock 10D:")
print(f"x = {best_solution}")
print(f"f(x) = {best_fitness:.10f}")
print(f"Distância ótimo: {np.linalg.norm(best_solution - 1):.6f}")

# Comparar com PSO/DE (executar PSO/DE no Rosenbrock)
from scipy.optimize import differential_evolution
de_result = differential_evolution(rosenbrock, [(-5, 5)]*10, maxiter=200, seed=42)

print(f"📊 Comparação Rosenbrock 10D:")
print(f"  CMA-ES: {best_fitness:.10f}")
print(f"  DE (scipy): {de_result.fun:.10f}")
print(f"  CMA-ES é {de_result.fun / best_fitness:.1f}x melhor")
