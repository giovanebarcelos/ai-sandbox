# GO1925-BenchmarkComparativoAlgoritmos
import time

algorithms = {
    'Random Search': lambda: random_search(rastrigin, bounds, n_samples=10000),
    'DE': lambda: differential_evolution(rastrigin, bounds, pop_size=100, max_iter=200),
    'PSO': lambda: pso.optimize(),
    'CMA-ES': lambda: cma_optimize(rastrigin, bounds)
}

results = {}
for name, algo in algorithms.items():
    start = time.time()
    best_sol, best_fit = algo()
    elapsed = time.time() - start
    results[name] = {'fitness': best_fit, 'time': elapsed}
    print(f"{name:20s}: Fitness={best_fit:.6f}, Time={elapsed:.2f}s")

# Típicos resultados Rastrigin 10D:
# Random Search:       Fitness=48.234567, Time=0.03s
# DE:                  Fitness=0.000234, Time=1.20s  ← Melhor
# PSO:                 Fitness=0.002456, Time=0.85s
# CMA-ES:              Fitness=0.000012, Time=1.50s  ← Mais preciso
