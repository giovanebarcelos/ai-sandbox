# GO1922A-ACORoteamentoLogistica
class ACO_VRP:
    """ACO para Vehicle Routing Problem (simplificado)"""
    def __init__(self, distances, demands, vehicle_capacity, n_vehicles):
        self.distances = distances
        self.demands = demands  # Demanda de cada cliente
        self.capacity = vehicle_capacity
        self.n_vehicles = n_vehicles
        self.n_clients = len(demands) - 1  # Cliente 0 = depósito

        # Feromônio e heurística (como ACO_TSP)
        self.pheromone = np.ones_like(distances)
        self.heuristic = 1 / (distances + 1e-10)

    def construct_routes(self):
        """Construir rotas para todos veículos"""
        routes = []
        remaining_clients = set(range(1, self.n_clients + 1))  # Excluir depósito 0

        for v in range(self.n_vehicles):
            route = [0]  # Começar no depósito
            current_load = 0
            current_pos = 0

            while remaining_clients:
                # Encontrar próximo cliente (probabilístico baseado em feromônio)
                feasible = [c for c in remaining_clients 
                            if current_load + self.demands[c] <= self.capacity]

                if not feasible:
                    break  # Veículo cheio, retornar depósito

                # Calcular probabilidades
                probs = np.array([
                    self.pheromone[current_pos, c] * self.heuristic[current_pos, c]
                    for c in feasible
                ])
                probs /= probs.sum()

                next_client = np.random.choice(feasible, p=probs)
                route.append(next_client)
                current_load += self.demands[next_client]
                current_pos = next_client
                remaining_clients.remove(next_client)

            route.append(0)  # Retornar ao depósito
            routes.append(route)

        return routes

    # ... (métodos de atualização feromônio similares a ACO_TSP)

# Exemplo VRP pequeno
distances_vrp = np.array([
    # Depósito e 5 clientes
    [0, 10, 15, 20, 25, 30],
    [10, 0, 8, 12, 18, 25],
    [15, 8, 0, 10, 15, 20],
    [20, 12, 10, 0, 8, 15],
    [25, 18, 15, 8, 0, 10],
    [30, 25, 20, 15, 10, 0]
])
demands = [0, 5, 3, 7, 4, 6]  # Depósito=0, clientes=5,3,7,4,6
vehicle_capacity = 10
n_vehicles = 3

# Rotas típicas ACO: 
# Veículo 1: [0, 2, 4, 0] (demanda 3+4=7)
# Veículo 2: [0, 1, 5, 0] (demanda 5+6=11 → infeasível, separar)
# Veículo 3: [0, 3, 0] (demanda 7)

print("🚚 Roteamento VRP com ACO:")
print("  Rotas otimizadas balanceando distância e capacidade")
