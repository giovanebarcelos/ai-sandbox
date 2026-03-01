# GO0202-IssoPermiteAoGoogleEntenderConexões
import networkx as nx
import matplotlib.pyplot as plt

# Criar grafo direcionado


if __name__ == "__main__":
    G = nx.DiGraph()

    # Adicionar nós (entidades)
    G.add_node("Einstein", tipo="Pessoa")
    G.add_node("Relatividade", tipo="Teoria")
    G.add_node("Nobel", tipo="Prêmio")
    G.add_node("Física", tipo="Área")
    G.add_node("1921", tipo="Ano")

    # Adicionar arestas (relações)
    G.add_edge("Einstein", "Relatividade", relacao="criou")
    G.add_edge("Einstein", "Nobel", relacao="ganhou")
    G.add_edge("Einstein", "Física", relacao="trabalha_em")
    G.add_edge("Nobel", "1921", relacao="ano")
    G.add_edge("Relatividade", "Física", relacao="pertence_a")

    # Consultas
    print("Vizinhos de Einstein:", list(G.neighbors("Einstein")))
    # ['Relatividade', 'Nobel', 'Física']

    print("\nCaminho Einstein → 1921:")
    caminho = nx.shortest_path(G, "Einstein", "1921")
    print(" → ".join(caminho))
    # Einstein → Nobel → 1921

    # Visualizar
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold',
            arrows=True, arrowsize=20, edge_color='gray')

    # Adicionar labels nas arestas
    edge_labels = nx.get_edge_attributes(G, 'relacao')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

    plt.title("Grafo de Conhecimento - Einstein")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
