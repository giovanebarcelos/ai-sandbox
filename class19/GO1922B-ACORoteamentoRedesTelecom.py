# GO1922B-ACORoteamentoRedesTelecom
class ACO_NetworkRouting:
    """ACO para roteamento adaptativo em redes"""
    def __init__(self, network_topology, traffic_matrix):
        """
        network_topology: grafo (nós=roteadores, arestas=links)
        traffic_matrix[src, dst]: volume tráfego (Gbps)
        """
        self.topology = network_topology
        self.traffic = traffic_matrix

        # Feromônio em cada link (representa "qualidade" do caminho)
        self.pheromone = {edge: 1.0 for edge in network_topology.edges()}

        # Heurística: preferir links baixa latência e alta banda
        self.heuristic = {
            edge: 1 / (topology[edge[0]][edge[1]]['latency'] + 1e-6)
            for edge in network_topology.edges()
        }

    def find_path(self, src, dst):
        """Formiga encontra caminho de src → dst"""
        path = [src]
        current = src

        while current != dst:
            neighbors = list(self.topology.neighbors(current))
            unvisited = [n for n in neighbors if n not in path]

            if not unvisited:
                return None  # Sem caminho

            # Probabilidade baseada em feromônio + heurística
            probs = []
            for neighbor in unvisited:
                edge = (current, neighbor)
                tau = self.pheromone.get(edge, 1.0)
                eta = self.heuristic.get(edge, 1.0)
                probs.append(tau * eta)

            probs = np.array(probs)
            probs /= probs.sum()

            next_node = np.random.choice(unvisited, p=probs)
            path.append(next_node)
            current = next_node

        return path

    def update_pheromone_by_latency(self, path, latency):
        """Reforçar feromônio em caminhos rápidos"""
        delta = 100 / latency  # Menor latência → mais feromônio

        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            self.pheromone[edge] = (1 - 0.1) * self.pheromone[edge] + delta

# Aplicação: Rede 10 roteadores, rotear 100 fluxos/segundo
# ACO adapta rotas dinamicamente baseado em congestionamento
print("🌐 Roteamento Rede com ACO:")
print("  Paths adaptativos minimizando latência e balanceando carga")
