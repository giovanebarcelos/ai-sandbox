# GO0307-24VisualizaçãoDoLabirintoESolução
import matplotlib.pyplot as plt
import numpy as np

def visualizar_labirinto(problema, caminho=None):
    """
    Visualiza labirinto e solução encontrada

    Args:
        problema: ProblemaLabirinto
        caminho: lista de estados (solução)
    """
    # Criar cópia do labirinto
    lab = np.array(problema.labirinto)

    # Marcar início e objetivo
    inicio_l, inicio_c = problema.estado_inicial
    obj_l, obj_c = problema.objetivo

    # Criar matriz de visualização
    visual = np.zeros((lab.shape[0], lab.shape[1], 3))

    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            if lab[i, j] == 1:  # Parede
                visual[i, j] = [0, 0, 0]  # Preto
            else:  # Caminho livre
                visual[i, j] = [1, 1, 1]  # Branco

    # Marcar início (verde)
    visual[inicio_l, inicio_c] = [0, 1, 0]

    # Marcar objetivo (vermelho)
    visual[obj_l, obj_c] = [1, 0, 0]

    # Marcar caminho (azul)
    if caminho:
        for estado in caminho[1:-1]:  # Exceto início e fim
            l, c = estado
            visual[l, c] = [0, 0.5, 1]

    # Plotar
    plt.figure(figsize=(10, 10))
    plt.imshow(visual)
    plt.title('Labirinto - Verde=Início, Vermelho=Objetivo, Azul=Caminho')
    plt.grid(True, color='gray', linewidth=0.5)
    plt.xticks(range(lab.shape[1]))
    plt.yticks(range(lab.shape[0]))
    plt.show()

# Exemplo de uso
labirinto = [
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]
]

problema = ProblemaLabirinto(
    labirinto=labirinto,
    inicio=(0, 0),
    objetivo=(9, 9)
)

# Resolver com A*
solucao = busca_a_estrela(problema, problema.heuristica_manhattan)

if solucao:
    # Reconstruir caminho
    caminho = [problema.estado_inicial]
    estado = problema.estado_inicial
    for acao in solucao:
        estado = problema.resultado(estado, acao)
        caminho.append(estado)

    visualizar_labirinto(problema, caminho)
