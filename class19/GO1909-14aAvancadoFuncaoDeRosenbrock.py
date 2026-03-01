# GO1909-14aAvançadoFunçãoDeRosenbrock
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ═══════════════════════════════════════════════════════════════════
# 1. DEFINIR FUNÇÃO DE ROSENBROCK
# ═══════════════════════════════════════════════════════════════════

def rosenbrock(x, y):
    """Função de Rosenbrock 2D"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def fitness_rosenbrock(individual):
    """
    Fitness = -f(x,y) pois queremos MAXIMIZAR fitness
    mas MINIMIZAR Rosenbrock
    """
    x, y = individual
    return -rosenbrock(x, y)

# ═══════════════════════════════════════════════════════════════════
# 2. VISUALIZAR SUPERFÍCIE ROSENBROCK
# ═══════════════════════════════════════════════════════════════════

x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

fig = plt.figure(figsize=(14, 6))

# Plot 2D contour
ax1 = fig.add_subplot(121)
contour = ax1.contourf(X, Y, Z, levels=np.logspace(-1, 3.5, 50), cmap='viridis')
ax1.plot(1, 1, 'r*', markersize=20, label='Mínimo Global (1,1)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Rosenbrock Function (Contour)')
ax1.legend()
plt.colorbar(contour, ax=ax1)

# Plot 3D surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
ax2.plot([1], [1], [0], 'r*', markersize=15, label='Ótimo')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('Rosenbrock Function (3D)')
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

print("🎯 Mínimo Global: (x, y) = (1, 1), f = 0")
print("🌋 Vale estreito em forma de banana dificulta convergência")

# ═══════════════════════════════════════════════════════════════════
# 3. ALGORITMO GENÉTICO PARA ROSENBROCK
# ═══════════════════════════════════════════════════════════════════

# Hiperparâmetros
POP_SIZE = 100
GENERATIONS = 200
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
BOUNDS = [(-2, 2), (-1, 3)]  # x ∈ [-2, 2], y ∈ [-1, 3]
N_ELITE = 2

# População inicial
def create_population_rosenbrock(pop_size, bounds):
    """Criar população com valores aleatórios nos limites"""
    population = []
    for _ in range(pop_size):
        individual = [np.random.uniform(low, high) for low, high in bounds]
        population.append(individual)
    return population

# Crossover BLX-α (Blend Crossover)
def blend_crossover(parent1, parent2, alpha=0.5):
    """Interpolar entre pais com expansão alpha"""
    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        min_g, max_g = min(g1, g2), max(g1, g2)
        range_g = max_g - min_g
        lower = min_g - alpha * range_g
        upper = max_g + alpha * range_g
        child1.append(np.random.uniform(lower, upper))
        child2.append(np.random.uniform(lower, upper))
    return child1, child2

# Mutação Gaussiana
def gaussian_mutation(individual, bounds, sigma=0.2):
    """Adicionar ruído gaussiano aos genes"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if np.random.random() < MUTATION_RATE:
            # Mutação gaussiana
            noise = np.random.normal(0, sigma)
            mutated[i] += noise
            # Garantir limites
            low, high = bounds[i]
            mutated[i] = np.clip(mutated[i], low, high)
    return mutated

# Seleção por torneio
def tournament_selection(population, fitness_values, k=5):
    """Selecionar melhores de k aleatórios"""
    selected = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), k, replace=False)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx].copy())
    return selected

# ═══════════════════════════════════════════════════════════════════
# 4. EXECUTAR ALGORITMO GENÉTICO
# ═══════════════════════════════════════════════════════════════════

### 📈 Exemplo de Evolução do AG

**PROBLEMA:** Maximizar f(x) = -x² + 10x  (parábola)

**DOMÍNIO:** x ∈ [0, 10]

**ÓTIMO TEÓRICO:** x* = 5, f(x*) = 25

**GERAÇÃO 0 (inicial aleatória):**
- x: [1.2, 8.7, 3.4, 9.1, 2.8, ...]
- fitness: [9.56, -13.69, 20.44, -17.81, 20.16, ...]
- MELHOR: x=3.4, f=20.44

**GERAÇÃO 10:**
- x: [4.2, 5.3, 4.8, 5.1, 4.5, ...]
- fitness: [24.36, 24.91, 24.96, 24.99, 24.75, ...]
- MELHOR: x=5.1, f=24.99  ← próximo do ótimo!

**GERAÇÃO 50:**
- x: [5.0, 5.0, 5.0, 5.0, 4.9, ...]
- fitness: [25.00, 25.00, 25.00, 25.00, 24.99, ...]
- MELHOR: x=5.0, f=25.00  ✔ CONVERGIU!

**EVOLUÇÃO FITNESS:**
- Gen 0:  melhor=20.44, média=8.23
- Gen 10: melhor=24.99, média=23.50
- Gen 50: melhor=25.00, média=24.98  ← população convergiu

![Gráfico mostrando evolução do fitness ao longo das gerações](images/Aula19_03.png)

population = create_population_rosenbrock(POP_SIZE, BOUNDS)
best_fitness_history = []
best_solution_history = []

print("\n" + "="*60)
print("OTIMIZANDO FUNÇÃO DE ROSENBROCK COM ALGORITMO GENÉTICO")
print("="*60)

for generation in range(GENERATIONS):
    # Avaliar população
    fitness_values = [fitness_rosenbrock(ind) for ind in population]

    # Rastrear melhor
    best_idx = np.argmax(fitness_values)
    best_individual = population[best_idx]
    best_fitness = fitness_values[best_idx]
    best_fitness_history.append(best_fitness)
    best_solution_history.append(best_individual.copy())

    # Log
    if generation % 20 == 0 or generation == GENERATIONS - 1:
        x_best, y_best = best_individual
        f_best = -best_fitness  # Converter para valor Rosenbrock
        print(f"Geração {generation:3d}: "
              f"f({x_best:.4f}, {y_best:.4f}) = {f_best:.6f}")

    # Seleção
    selected = tournament_selection(population, fitness_values, k=5)

    # Crossover
    offspring = []
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected) and np.random.random() < CROSSOVER_RATE:
            child1, child2 = blend_crossover(selected[i], selected[i+1], alpha=0.5)
            offspring.extend([child1, child2])
        else:
            offspring.extend([selected[i], selected[i+1] if i+1 < len(selected) else selected[i]])

    # Mutação
    offspring = [gaussian_mutation(ind, BOUNDS, sigma=0.2) for ind in offspring]

    # Elitismo
    elite_indices = np.argsort(fitness_values)[-N_ELITE:]
    for i, elite_idx in enumerate(elite_indices):
        offspring[i] = population[elite_idx].copy()

    population = offspring

# ═══════════════════════════════════════════════════════════════════
# 5. RESULTADOS FINAIS
# ═══════════════════════════════════════════════════════════════════

x_final, y_final = best_solution_history[-1]
f_final = -best_fitness_history[-1]

print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print(f"🏆 Melhor Solução: (x, y) = ({x_final:.6f}, {y_final:.6f})")
print(f"📊 f(x, y) = {f_final:.6f}")
print(f"🎯 Ótimo Global: (x, y) = (1, 1), f = 0")
print(f"📉 Erro: {abs(f_final - 0):.6f}")
print(f"📍 Distância do ótimo: {np.sqrt((x_final-1)**2 + (y_final-1)**2):.6f}")

# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZAR CONVERGÊNCIA
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Evolução do fitness
ax1 = axes[0]
rosenbrock_values = [-f for f in best_fitness_history]
ax1.plot(rosenbrock_values, linewidth=2)
ax1.axhline(y=0, color='r', linestyle='--', label='Ótimo Global')
ax1.set_xlabel('Geração')
ax1.set_ylabel('f(x, y)')
ax1.set_title('Convergência - Valor Função Rosenbrock')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Trajetória no espaço de busca
ax2 = axes[1]
contour = ax2.contourf(X, Y, Z, levels=np.logspace(-1, 3.5, 50), 
                        cmap='viridis', alpha=0.6)
x_traj = [sol[0] for sol in best_solution_history[::5]]
y_traj = [sol[1] for sol in best_solution_history[::5]]
ax2.plot(x_traj, y_traj, 'ro-', linewidth=2, markersize=4, 
         label='Trajetória GA', alpha=0.7)
ax2.plot(x_traj[0], y_traj[0], 'go', markersize=10, label='Início')
ax2.plot(x_traj[-1], y_traj[-1], 'bo', markersize=10, label='Final')
ax2.plot(1, 1, 'r*', markersize=20, label='Ótimo (1,1)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Trajetória no Espaço de Busca')
ax2.legend()
plt.colorbar(contour, ax=ax2, label='f(x,y)')

plt.tight_layout()
plt.show()

print("\n💡 Observações:")
print("   • AG encontra vale rapidamente mas converge lentamente dentro dele")
print("   • Crossover BLX-α ajuda exploração inicial")
print("   • Mutação gaussiana refina solução no vale estreito")
print("   • Elitismo garante não perder boas soluções")
