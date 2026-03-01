# GO1920-Slides1822LoopPrincipalAg
# Algoritmo Genético para TSP
POP_SIZE = 200
GENERATIONS = 500
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2

population = create_population_tsp(POP_SIZE, N_CITIES)
best_distance_history = []

for generation in range(GENERATIONS):
    # Avaliar
    fitness_values = [fitness_tsp(ind, cities) for ind in population]
    distances = [-f for f in fitness_values]  # Converter para distâncias positivas

    best_distance = min(distances)
    best_distance_history.append(best_distance)

    if generation % 50 == 0:
        print(f"Geração {generation}: Melhor distância = {best_distance:.2f}")

    # Seleção
    selected = tournament_selection_tsp(population, cities, k=5)

    # Crossover
    offspring = []
    for i in range(0, len(selected), 2):
        if i + 1 < len(selected) and np.random.random() < CROSSOVER_RATE:
            child1 = order_crossover(selected[i], selected[i+1])
            child2 = order_crossover(selected[i+1], selected[i])
            offspring.extend([child1, child2])
        else:
            offspring.extend([selected[i], selected[i+1] if i+1 < len(selected) else selected[i]])

    # Mutação
    offspring = [swap_mutation_tsp(ind, MUTATION_RATE) for ind in offspring]

    # Elitismo: manter melhor
    best_idx = np.argmax(fitness_values)
    offspring[0] = population[best_idx].copy()

    population = offspring

# Melhor rota final
fitness_values = [fitness_tsp(ind, cities) for ind in population]
best_idx = np.argmax(fitness_values)
best_route = population[best_idx]
best_distance = -fitness_values[best_idx]

print(f"\n🏆 MELHOR ROTA TSP:")
print(f"Distância total: {best_distance:.2f}")
print(f"Rota: {best_route}")

# Plotar evolução
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(best_distance_history)
plt.xlabel('Geração')
plt.ylabel('Melhor Distância')
plt.title('Evolução da Melhor Distância')
plt.grid(True)

# Plotar melhor rota
plt.subplot(1, 2, 2)
route_cities = [cities[i] for i in best_route]
route_cities.append(cities[best_route[0]])  # Fechar ciclo
route_cities = np.array(route_cities)

plt.plot(route_cities[:, 0], route_cities[:, 1], 'o-', linewidth=2, markersize=8)
plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=5)
for i, city_idx in enumerate(best_route):
    plt.text(cities[city_idx, 0], cities[city_idx, 1], str(city_idx), 
             fontsize=9, ha='center', va='center')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Melhor Rota (Distância: {best_distance:.2f})')
plt.grid(True)
plt.tight_layout()
plt.show()
