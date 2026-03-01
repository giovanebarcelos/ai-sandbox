# GO1922-AntColonyOptimizationCompletaTSP
import numpy as np
import matplotlib.pyplot as plt

class ACO_TSP:
    def __init__(self, distances, n_ants=20, n_iterations=100, 
                 alpha=1.0, beta=2.0, rho=0.1, Q=100, seed=42):
        """
        ACO para Travelling Salesman Problem

        Args:
            distances: matriz distâncias [n_cities x n_cities]
            n_ants: número de formigas por iteração
            alpha: peso feromônio (típico: 1.0)
            beta: peso heurística (típico: 2-5)
            rho: taxa evaporação (típico: 0.1-0.3)
            Q: constante deposição feromônio
        """
        self.distances = distances
        self.n_cities = len(distances)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho  # Evaporação
        self.Q = Q

        np.random.seed(seed)

        # Inicializar feromônio (igual em todas arestas)
        self.pheromone = np.ones((self.n_cities, self.n_cities))

        # Heurística: inverso da distância (prefere cidades próximas)
        self.heuristic = 1 / (distances + 1e-10)
        np.fill_diagonal(self.heuristic, 0)

        # Rastreamento
        self.best_tour = None
        self.best_length = np.inf
        self.history = []

    def construct_solution(self):
        """Uma formiga constrói um tour completo"""
        # Começar de cidade aleatória
        tour = [np.random.randint(self.n_cities)]
        visited = set(tour)

        while len(tour) < self.n_cities:
            current = tour[-1]

            # Calcular probabilidades para próxima cidade
            unvisited = [c for c in range(self.n_cities) if c not in visited]

            if not unvisited:
                break

            # Probabilidade ∝ τ^α * η^β
            probs = np.zeros(len(unvisited))
            for idx, city in enumerate(unvisited):
                probs[idx] = (self.pheromone[current, city] ** self.alpha *
                              self.heuristic[current, city] ** self.beta)

            # Normalizar
            probs /= probs.sum()

            # Escolher próxima cidade
            next_city = np.random.choice(unvisited, p=probs)
            tour.append(next_city)
            visited.add(next_city)

        return tour

    def tour_length(self, tour):
        """Calcular comprimento total do tour"""
        length = sum(self.distances[tour[i], tour[i+1]] 
                     for i in range(len(tour) - 1))
        length += self.distances[tour[-1], tour[0]]  # Voltar ao início
        return length

    def update_pheromone(self, all_tours, all_lengths):
        """Atualizar feromônio: evaporação + deposição"""
        # 1. EVAPORAÇÃO
        self.pheromone *= (1 - self.rho)

        # 2. DEPOSIÇÃO (apenas melhores formigas)
        best_idx = np.argmin(all_lengths)
        best_tour = all_tours[best_idx]
        best_length = all_lengths[best_idx]

        # Adicionar feromônio na melhor rota
        delta = self.Q / best_length
        for i in range(len(best_tour) - 1):
            self.pheromone[best_tour[i], best_tour[i+1]] += delta
            self.pheromone[best_tour[i+1], best_tour[i]] += delta  # Simétrico

        # Fechar ciclo
        self.pheromone[best_tour[-1], best_tour[0]] += delta
        self.pheromone[best_tour[0], best_tour[-1]] += delta

    def optimize(self):
        """Executar ACO"""
        for iteration in range(self.n_iterations):
            # Construir soluções (todas as formigas)
            all_tours = [self.construct_solution() for _ in range(self.n_ants)]
            all_lengths = [self.tour_length(tour) for tour in all_tours]

            # Atualizar melhor solução
            iter_best_idx = np.argmin(all_lengths)
            iter_best_length = all_lengths[iter_best_idx]

            if iter_best_length < self.best_length:
                self.best_length = iter_best_length
                self.best_tour = all_tours[iter_best_idx]

            # Atualizar feromônio
            self.update_pheromone(all_tours, all_lengths)

            # Histórico
            self.history.append(self.best_length)

            if iteration % 20 == 0:
                print(f"Iter {iteration:3d}: Best length = {self.best_length:.2f}")

        return self.best_tour, self.best_length


# EXEMPLO: TSP 20 cidades
np.random.seed(42)
n_cities = 20
coords = np.random.rand(n_cities, 2) * 100  # Coordenadas [0, 100]

# Calcular matriz distâncias
distances = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        distances[i, j] = np.linalg.norm(coords[i] - coords[j])

# Otimizar com ACO
aco = ACO_TSP(distances, n_ants=30, n_iterations=200, alpha=1.0, beta=3.0, rho=0.1, Q=100)
best_tour, best_length = aco.optimize()

print(f"\n🐜 ACO - Resultado TSP:")
print(f"  Melhor tour: {best_tour}")
print(f"  Comprimento: {best_length:.2f}")

# Comparar com solução gulosa (nearest neighbor)
def nearest_neighbor(distances):
    tour = [0]
    visited = {0}
    for _ in range(len(distances) - 1):
        current = tour[-1]
        nearest = min((d, c) for c, d in enumerate(distances[current]) 
                      if c not in visited)[1]
        tour.append(nearest)
        visited.add(nearest)
    return tour

greedy_tour = nearest_neighbor(distances)
greedy_length = aco.tour_length(greedy_tour)

print(f"📊 Comparação:")
print(f"  ACO: {best_length:.2f}")
print(f"  Greedy: {greedy_length:.2f}")
print(f"  Melhoria: {(greedy_length - best_length) / greedy_length * 100:.1f}%")

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Convergência
ax1.plot(aco.history, linewidth=2, color='brown')
ax1.axhline(greedy_length, color='green', linestyle='--', label='Greedy')
ax1.set_xlabel('Iteração')
ax1.set_ylabel('Comprimento Tour')
ax1.set_title('Convergência ACO')
ax1.legend()
ax1.grid(alpha=0.3)

# Tour ótimo
ax2.scatter(coords[:, 0], coords[:, 1], s=200, c='red', zorder=3)
for i, (x, y) in enumerate(coords):
    ax2.text(x, y, str(i), ha='center', va='center', fontsize=10, color='white', weight='bold')

# Desenhar tour
tour_coords = coords[best_tour + [best_tour[0]]]
ax2.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=2, alpha=0.6)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Melhor Tour ACO (Length={best_length:.2f})')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
