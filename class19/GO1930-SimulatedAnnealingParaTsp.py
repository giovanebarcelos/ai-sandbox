# GO1930-SimulatedAnnealingParaTsp
def simulated_annealing_tsp(dist_matrix, max_iterations=10000, 
                            initial_temp=100, cooling_rate=0.995):
    """Simulated Annealing para TSP"""
    n_cities = len(dist_matrix)

    # Solução inicial aleatória
    current_solution = np.random.permutation(n_cities)
    current_distance = calculate_total_distance(current_solution, dist_matrix)

    best_solution = current_solution.copy()
    best_distance = current_distance

    temperature = initial_temp
    history = []

    for iteration in range(max_iterations):
        # Gerar vizinho (2-opt swap)
        i, j = sorted(np.random.choice(n_cities, 2, replace=False))
        neighbor = current_solution.copy()
        neighbor[i:j+1] = neighbor[i:j+1][::-1]  # Reverter segmento

        neighbor_distance = calculate_total_distance(neighbor, dist_matrix)

        # Calcular diferença
        delta = neighbor_distance - current_distance

        # Aceitar solução melhor OU pior com probabilidade e^(-delta/T)
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            current_solution = neighbor
            current_distance = neighbor_distance

            # Atualizar melhor
            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance

        # Resfriamento
        temperature *= cooling_rate
        history.append(best_distance)

        if (iteration + 1) % 1000 == 0:
            print(f"Iteração {iteration+1}: Melhor = {best_distance:.2f}, Temp = {temperature:.2f}")

    return best_solution, history

# Executar SA
print("\n🔥 Resolvendo TSP com Simulated Annealing...\n")
sa_route, sa_history = simulated_annealing_tsp(dist_matrix, max_iterations=10000)

print(f"\n✅ Melhor rota SA: {sa_route}")
print(f"✅ Distância SA: {calculate_total_distance(sa_route, dist_matrix):.2f}")

# Comparar AG vs SA
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history, label='AG', linewidth=2)
plt.plot(sa_history, label='SA', linewidth=2, alpha=0.7)
plt.xlabel('Iterações', fontsize=12)
plt.ylabel('Melhor Distância', fontsize=12)
plt.title('AG vs Simulated Annealing - Convergência', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Rotas lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# AG route
route_ag = np.append(best_route, best_route[0])
ax1.plot(cities[route_ag, 0], cities[route_ag, 1], 'o-', linewidth=2, markersize=8)
ax1.set_title(f'AG: Distância = {calculate_total_distance(best_route, dist_matrix):.2f}', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# SA route
route_sa = np.append(sa_route, sa_route[0])
ax2.plot(cities[route_sa, 0], cities[route_sa, 1], 'o-', linewidth=2, markersize=8, color='orange')
ax2.set_title(f'SA: Distância = {calculate_total_distance(sa_route, dist_matrix):.2f}', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Estatísticas comparativas
print("\n📊 Comparação AG vs SA:")
print(f"  AG: {calculate_total_distance(best_route, dist_matrix):.2f}")
print(f"  SA: {calculate_total_distance(sa_route, dist_matrix):.2f}")
print(f"  Diferença: {abs(calculate_total_distance(best_route, dist_matrix) - calculate_total_distance(sa_route, dist_matrix)):.2f}")
