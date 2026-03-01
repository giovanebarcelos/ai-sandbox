# GO0312-BuscaDijkstraGrafoPonderado
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

print("="*70)
print("ALGORITMO DE DIJKSTRA - BUSCA DE CAMINHO MAIS CURTO")
print("="*70)

class GrafoCidades:
    """Grafo ponderado representando cidades e distâncias"""

    def __init__(self):
        self.grafo = {}  # {cidade: [(vizinho, distancia), ...]}

    def adicionar_aresta(self, origem: str, destino: str, distancia: float):
        """Adiciona aresta bidirecional com peso (distância)"""
        if origem not in self.grafo:
            self.grafo[origem] = []
        if destino not in self.grafo:
            self.grafo[destino] = []

        self.grafo[origem].append((destino, distancia))
        self.grafo[destino].append((origem, distancia))

    def vizinhos(self, cidade: str) -> List[Tuple[str, float]]:
        """Retorna vizinhos e distâncias"""
        return self.grafo.get(cidade, [])

    def dijkstra(self, origem: str, destino: str) -> Tuple[Optional[List[str]], float, Dict]:
        """
        Implementação do algoritmo de Dijkstra

        Returns:
            (caminho, custo_total, estatisticas)
        """
        # Inicialização
        distancias = {cidade: float('inf') for cidade in self.grafo}
        distancias[origem] = 0
        anteriores = {cidade: None for cidade in self.grafo}
        visitados = set()

        # Fila de prioridade: (distancia, cidade)
        fila = [(0, origem)]
        nos_explorados = 0
        max_fila = 1

        while fila:
            max_fila = max(max_fila, len(fila))
            dist_atual, cidade_atual = heapq.heappop(fila)

            # Se já visitado, pular
            if cidade_atual in visitados:
                continue

            visitados.add(cidade_atual)
            nos_explorados += 1

            # Se chegou no destino, reconstruir caminho
            if cidade_atual == destino:
                caminho = []
                while cidade_atual:
                    caminho.append(cidade_atual)
                    cidade_atual = anteriores[cidade_atual]
                caminho.reverse()

                estatisticas = {
                    'nos_explorados': nos_explorados,
                    'nos_visitados': len(visitados),
                    'max_fila': max_fila
                }

                return caminho, distancias[destino], estatisticas

            # Explorar vizinhos
            for vizinho, distancia in self.vizinhos(cidade_atual):
                if vizinho in visitados:
                    continue

                nova_distancia = distancias[cidade_atual] + distancia

                if nova_distancia < distancias[vizinho]:
                    distancias[vizinho] = nova_distancia
                    anteriores[vizinho] = cidade_atual
                    heapq.heappush(fila, (nova_distancia, vizinho))

        # Sem caminho
        return None, float('inf'), {'nos_explorados': nos_explorados}

    def visualizar(self, caminho=None, titulo="Grafo de Cidades"):
        """Visualiza o grafo com destaque no caminho encontrado"""
        G = nx.Graph()

        # Adicionar arestas
        for cidade, vizinhos in self.grafo.items():
            for vizinho, dist in vizinhos:
                G.add_edge(cidade, vizinho, weight=dist)

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.9)

        # Destacar caminho se fornecido
        if caminho:
            # Nós do caminho em verde
            nx.draw_networkx_nodes(G, pos, nodelist=caminho, 
                                  node_color='lightgreen', node_size=2500)
            # Origem em verde escuro
            nx.draw_networkx_nodes(G, pos, nodelist=[caminho[0]], 
                                  node_color='darkgreen', node_size=2500)
            # Destino em vermelho
            nx.draw_networkx_nodes(G, pos, nodelist=[caminho[-1]], 
                                  node_color='red', node_size=2500)

            # Arestas do caminho em verde grosso
            caminho_edges = [(caminho[i], caminho[i+1]) for i in range(len(caminho)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=caminho_edges, 
                                  edge_color='green', width=4)

        # Desenhar todas as arestas
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

        # Labels das arestas (distâncias)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f"{v}km" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        plt.title(titulo, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# ═══════════════════════════════════════════════════════════════════
# CRIAR MAPA DE CIDADES DO RIO GRANDE DO SUL
# ═══════════════════════════════════════════════════════════════════

print("\n🗺️  CRIANDO MAPA DE CIDADES...")

mapa = GrafoCidades()

# Adicionar conexões (simplificado)
mapa.adicionar_aresta("Porto Alegre", "Canoas", 15)
mapa.adicionar_aresta("Porto Alegre", "Viamão", 25)
mapa.adicionar_aresta("Porto Alegre", "Gravataí", 30)
mapa.adicionar_aresta("Canoas", "Novo Hamburgo", 20)
mapa.adicionar_aresta("Canoas", "São Leopoldo", 12)
mapa.adicionar_aresta("Novo Hamburgo", "São Leopoldo", 8)
mapa.adicionar_aresta("Novo Hamburgo", "Campo Bom", 10)
mapa.adicionar_aresta("São Leopoldo", "Sapucaia do Sul", 15)
mapa.adicionar_aresta("Gravataí", "Cachoeirinha", 12)
mapa.adicionar_aresta("Cachoeirinha", "Canoas", 10)
mapa.adicionar_aresta("Campo Bom", "Dois Irmãos", 18)
mapa.adicionar_aresta("Dois Irmãos", "Ivoti", 12)
mapa.adicionar_aresta("Ivoti", "Estância Velha", 20)
mapa.adicionar_aresta("Viamão", "Alvorada", 18)
mapa.adicionar_aresta("Alvorada", "Cachoeirinha", 25)

print(f"✅ Mapa criado: {len(mapa.grafo)} cidades")

# ═══════════════════════════════════════════════════════════════════
# EXECUTAR BUSCA DE DIJKSTRA
# ═══════════════════════════════════════════════════════════════════

origem = "Porto Alegre"
destino = "Estância Velha"

print(f"\n🚗 BUSCANDO ROTA: {origem} → {destino}")
print("-" * 70)

caminho, distancia, stats = mapa.dijkstra(origem, destino)

if caminho:
    print(f"\n✅ CAMINHO ENCONTRADO!")
    print(f"   Rota: {' → '.join(caminho)}")
    print(f"   Distância total: {distancia:.1f} km")
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"   Nós explorados: {stats['nos_explorados']}")
    print(f"   Nós visitados: {stats['nos_visitados']}")
    print(f"   Fila máxima: {stats['max_fila']}")

    print(f"\n📍 DETALHAMENTO DA ROTA:")
    dist_acum = 0
    for i in range(len(caminho)-1):
        cidade1, cidade2 = caminho[i], caminho[i+1]
        # Encontrar distância
        for viz, dist in mapa.vizinhos(cidade1):
            if viz == cidade2:
                dist_acum += dist
                print(f"   {i+1}. {cidade1} → {cidade2}: {dist}km (acum: {dist_acum}km)")
                break

    # Visualizar
    print(f"\n🗺️  Gerando mapa visual...")
    mapa.visualizar(caminho, f"Rota Ótima: {origem} → {destino} ({distancia:.1f}km)")
else:
    print(f"❌ Nenhum caminho encontrado!")

# ═══════════════════════════════════════════════════════════════════
# COMPARAÇÃO: VÁRIAS ROTAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPARAÇÃO DE ROTAS ALTERNATIVAS")
print("="*70)

destinos_teste = ["Canoas", "Novo Hamburgo", "Ivoti", "Estância Velha"]

for dest in destinos_teste:
    caminho, dist, _ = mapa.dijkstra(origem, dest)
    if caminho:
        print(f"\n{origem} → {dest}:")
        print(f"  Rota: {' → '.join(caminho)}")
        print(f"  Distância: {dist:.1f} km")
