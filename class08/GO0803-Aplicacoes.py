# GO0803-Aplicações
# IMPLEMENTAR U-MATRIX

def calculate_umatrix(som):
    """
    Calcula U-Matrix (distâncias médias aos vizinhos)
    """
    umatrix = np.zeros((som.map_height, som.map_width))

    for i in range(som.map_height):
        for j in range(som.map_width):
            # Pegar pesos do neurônio atual
            w = som.weights[i, j]

            # Calcular distâncias aos vizinhos
            distances = []

            # Vizinhos: cima, baixo, esquerda, direita
            neighbors = [
                (i-1, j), (i+1, j), (i, j-1), (i, j+1)
            ]

            for ni, nj in neighbors:
                # Verificar se vizinho existe
                if 0 <= ni < som.map_height and 0 <= nj < som.map_width:
                    w_neighbor = som.weights[ni, nj]
                    dist = np.linalg.norm(w - w_neighbor)
                    distances.append(dist)

            # Média das distâncias
            umatrix[i, j] = np.mean(distances)

    return umatrix

# Calcular U-Matrix


if __name__ == "__main__":
    umatrix = calculate_umatrix(som)

    # Plotar
    plt.figure(figsize=(10, 8))
    plt.imshow(umatrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Distância Média aos Vizinhos')
    plt.title('U-Matrix: Estrutura de Clusters no SOM', fontsize=14)
    plt.xlabel('Posição X na Grade')
    plt.ylabel('Posição Y na Grade')
    plt.show()

    print("U-Matrix: Regiões escuras = clusters, claras = fronteiras")
