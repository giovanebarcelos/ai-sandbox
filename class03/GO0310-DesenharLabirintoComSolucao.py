# GO0310-DesenharLabirintoComSolução
import matplotlib.pyplot as plt
import numpy as np

def visualizar_solucao(problema, caminho):
    """Plota labirinto com caminho encontrado"""
    lab = np.array(problema.labirinto)
    visual = np.ones((lab.shape[0], lab.shape[1], 3))  # Branco por padrão

    # 1. Paredes = preto
    visual[lab == 1] = [0, 0, 0]

    # 2. Caminho = azul claro
    for estado in caminho[1:-1]:  # Exceto início e fim
        visual[estado[0], estado[1]] = [0.3, 0.5, 1.0]

    # 3. Início = verde
    inicio = problema.estado_inicial
    visual[inicio[0], inicio[1]] = [0, 1, 0]

    # 4. Objetivo = vermelho
    objetivo = problema.objetivo
    visual[objetivo[0], objetivo[1]] = [1, 0, 0]

    plt.figure(figsize=(8, 8))
    plt.imshow(visual)
    plt.title(f'Solução A* - {len(caminho)} passos, custo={len(caminho)-1}')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(lab.shape[1]))
    plt.yticks(range(lab.shape[0]))
    plt.show()

# Exemplo: usar com labirinto 5x5 simples
lab_simples = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]
problema_viz = ProblemaLabirinto(lab_simples, inicio=(0, 0), objetivo=(4, 4))
caminho_viz, _ = busca_a_estrela(problema_viz)
if caminho_viz:
    print(f"✅ Caminho: {caminho_viz}")
    visualizar_solucao(problema_viz, caminho_viz)
