# GO1920-DifferentialEvolutionDoZero
import numpy as np
import matplotlib.pyplot as plt

def differential_evolution(objective_func, bounds, pop_size=50, max_iter=200, 
                            F=0.8, CR=0.9, seed=42):
    """
    DE clássico (DE/rand/1/bin)

    Args:
        objective_func: função a MINIMIZAR (aceita array shape [n_dim])
        bounds: lista [(min1, max1), (min2, max2), ...] para cada dimensão
        pop_size: tamanho população (recomendado: 10 * n_dim)
        F: fator mutação (0.5-1.0) - exploração
        CR: crossover rate (0.7-0.9) - diversidade
    """
    np.random.seed(seed)
    n_dim = len(bounds)

    # Inicializar população aleatória
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], 
                             (pop_size, n_dim))
    fitness = np.array([objective_func(ind) for ind in pop])

    # Rastrear convergência
    best_history = []

    for generation in range(max_iter):
        for i in range(pop_size):
            # 1. MUTAÇÃO: escolher 3 indivíduos aleatórios (r1 ≠ r2 ≠ r3 ≠ i)
            indices = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)

            # Vetor mutante: v = x_r1 + F * (x_r2 - x_r3)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])

            # Garantir limites
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])

            # 2. CROSSOVER: binomial (mistura genes pai vs mutante)
            trial = pop[i].copy()
            j_rand = np.random.randint(n_dim)  # Garante pelo menos 1 gene mutado

            for j in range(n_dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # 3. SELEÇÃO: greedy (mantém melhor)
            trial_fitness = objective_func(trial)
            if trial_fitness < fitness[i]:  # Minimização
                pop[i] = trial
                fitness[i] = trial_fitness

        # Rastrear melhor solução
        best_idx = np.argmin(fitness)
        best_history.append(fitness[best_idx])

        if generation % 50 == 0:
            print(f"Gen {generation:3d}: Best fitness = {fitness[best_idx]:.6f}")

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], best_history


# EXEMPLO: Função Rastrigin (multimodal, difícil)
def rastrigin(x):
    """f(x) = 10n + Σ(x_i^2 - 10*cos(2π*x_i))
    Mínimo global: f(0, ..., 0) = 0
    """
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Otimizar 10 dimensões
bounds = [(-5.12, 5.12)] * 10
best_solution, best_fitness, history = differential_evolution(
    rastrigin, bounds, pop_size=100, max_iter=300, F=0.8, CR=0.9
)

print(f"\n🎯 Melhor solução encontrada:")
print(f"x = {best_solution}")
print(f"f(x) = {best_fitness:.6f}")
print(f"Distância do ótimo: {np.linalg.norm(best_solution):.4f}")

# Visualizar convergência
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history, linewidth=2)
plt.xlabel('Geração')
plt.ylabel('Melhor Fitness')
plt.title('Convergência DE - Rastrigin 10D')
plt.yscale('log')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
# Projetar solução em 2D (primeiras 2 dimensões)
x = np.linspace(-5.12, 5.12, 200)
y = np.linspace(-5.12, 5.12, 200)
X, Y = np.meshgrid(x, y)
Z = np.array([[rastrigin([xi, yi] + [0]*8) for xi, yi in zip(xrow, yrow)] 
               for xrow, yrow in zip(X, Y)])

plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Fitness')
plt.plot(best_solution[0], best_solution[1], 'r*', markersize=20, 
         label=f'DE: {best_fitness:.2f}')
plt.plot(0, 0, 'w*', markersize=15, label='Ótimo global')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Função Rastrigin (2D projection)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
