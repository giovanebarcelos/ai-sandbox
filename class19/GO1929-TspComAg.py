# GO1929-TspComAg
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Gerar cidades aleatórias
np.random.seed(42)
n_cities = 20
cities = np.random.rand(n_cities, 2) * 100  # Coordenadas (x, y)

# Matriz de distâncias
dist_matrix = distance_matrix(cities, cities)

def calculate_total_distance(route, dist_matrix):
    """Calcula distância total da rota"""
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i], route[i + 1]]
    # Retornar à origem
    distance += dist_matrix[route[-1], route[0]]
    return distance

def fitness_tsp(route, dist_matrix):
    """Fitness = 1 / distância (quanto menor distância, maior fitness)"""
    return 1.0 / calculate_total_distance(route, dist_matrix)

def create_population_tsp(pop_size, n_cities):
    """Cria população inicial (permutações aleatórias)"""
    population = []
    for _ in range(pop_size):
        route = np.random.permutation(n_cities)
        population.append(route)
    return population

def order_crossover(parent1, parent2):
    """Order Crossover (OX) - mantém ordem relativa"""
    size = len(parent1)
    child = [-1] * size

    # Selecionar segmento aleatório do parent1
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child[start:end+1] = parent1[start:end+1]

    # Preencher resto com genes do parent2 (na ordem)
    parent2_genes = [gene for gene in parent2 if gene not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = parent2_genes[idx]
            idx += 1

    return np.array(child)

def swap_mutation_tsp(route, mutation_rate=0.01):
    """Mutação por troca de 2 cidades"""
    route = route.copy()
    for i in range(len(route)):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(0, len(route))
            route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm_tsp(dist_matrix, pop_size=100, generations=500, 
                          elite_size=20, mutation_rate=0.01):
    """AG para TSP"""
    n_cities = len(dist_matrix)
    population = create_population_tsp(pop_size, n_cities)

    best_distances = []

    for gen in range(generations):
        # Avaliar fitness
        fitness_scores = [fitness_tsp(ind, dist_matrix) for ind in population]

        # Registrar melhor
        best_idx = np.argmax(fitness_scores)
        best_distance = calculate_total_distance(population[best_idx], dist_matrix)
        best_distances.append(best_distance)

        # Seleção por torneio
        selected = []
        for _ in range(pop_size - elite_size):
            tournament = np.random.choice(pop_size, 3, replace=False)
            winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
            selected.append(population[winner])

        # Elitismo (manter melhores)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite = [population[i] for i in elite_indices]

        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1, parent2 = selected[i], selected[i + 1]
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
                offspring.extend([child1, child2])

        # Mutação
        offspring = [swap_mutation_tsp(child, mutation_rate) for child in offspring]

        # Nova população
        population = elite + offspring[:pop_size - elite_size]

        if (gen + 1) % 50 == 0:
            print(f"Geração {gen+1}: Melhor distância = {best_distance:.2f}")

    # Retornar melhor solução
    fitness_scores = [fitness_tsp(ind, dist_matrix) for ind in population]
    best_idx = np.argmax(fitness_scores)
    best_route = population[best_idx]

    return best_route, best_distances

# Executar AG
print("🧬 Resolvendo TSP com 20 cidades...\n")
best_route, history = genetic_algorithm_tsp(dist_matrix, pop_size=100, 
                                            generations=500, mutation_rate=0.02)

print(f"\n✅ Melhor rota encontrada: {best_route}")
print(f"✅ Distância total: {calculate_total_distance(best_route, dist_matrix):.2f}")

# Visualizar convergência
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history, linewidth=2)
plt.xlabel('Geração', fontsize=12)
plt.ylabel('Melhor Distância', fontsize=12)
plt.title('Convergência do AG para TSP', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Visualizar melhor rota
plt.subplot(1, 2, 2)
route_cities = np.append(best_route, best_route[0])  # Fechar ciclo
plt.plot(cities[route_cities, 0], cities[route_cities, 1], 
         'o-', linewidth=2, markersize=10, label='Rota otimizada')
plt.scatter(cities[0, 0], cities[0, 1], color='red', s=200, 
            marker='*', zorder=5, label='Início/Fim')

for i, (x, y) in enumerate(cities):
    plt.text(x + 1, y + 1, str(i), fontsize=9)

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Melhor Rota Encontrada', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📊 Análise:")
print(f"  • Melhoria: {history[0]:.2f} → {history[-1]:.2f} ({((history[0]-history[-1])/history[0]*100):.1f}% redução)")
print(f"  • Convergiu em ~{np.argmin(np.diff(history[-50:])) + len(history) - 50} gerações")
