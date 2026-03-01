# GO0802-8DeUsoCoresRgb
# EXEMPLO: ORGANIZAR CORES RGB COM SOM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# GERAR CORES ALEATÓRIAS

np.random.seed(42)

# 1000 cores RGB aleatórias (valores entre 0 e 1)
n_colors = 1000
colors = np.random.rand(n_colors, 3)

print("="*60)
print("TREINANDO SOM PARA ORGANIZAR CORES")
print("="*60)
print(f"Dados: {n_colors} cores RGB (3 dimensões)")
print(f"Grade SOM: 20×20 = 400 neurônios")
print(f"Iterações: 1000")

# TREINAR SOM

som = SimpleSOM(map_height=20, map_width=20, input_dim=3,
               learning_rate=0.5, n_iterations=1000)

som.fit(colors, verbose=False)

print(f"\nErro de quantização: {som.quantization_error(colors):.4f}")

# VISUALIZAR MAPA DE CORES

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Mapa de cores organizado
ax1 = axes[0]
for i in range(som.map_height):
    for j in range(som.map_width):
        # Pegar cor do neurônio (pesos são RGB)
        color = som.weights[i, j]
        # Garantir que está no intervalo [0, 1]
        color = np.clip(color, 0, 1)
        # Desenhar retângulo
        rect = Rectangle((j, som.map_height - i - 1), 1, 1, 
                        facecolor=color, edgecolor='none')
        ax1.add_patch(rect)

ax1.set_xlim(0, som.map_width)
ax1.set_ylim(0, som.map_height)
ax1.set_aspect('equal')
ax1.set_title('Mapa Auto-Organizado de Cores RGB', fontsize=14)
ax1.set_xlabel('Posição X na Grade')
ax1.set_ylabel('Posição Y na Grade')
ax1.grid(True, alpha=0.3)

# Plot 2: Cores originais (amostra)
ax2 = axes[1]
sample_colors = colors[:400]  # Amostra de 400 cores
for idx, color in enumerate(sample_colors):
    i, j = idx // 20, idx % 20
    rect = Rectangle((j, 20 - i - 1), 1, 1, 
                    facecolor=color, edgecolor='none')
    ax2.add_patch(rect)

ax2.set_xlim(0, 20)
ax2.set_ylim(0, 20)
ax2.set_aspect('equal')
ax2.set_title('Amostra de Cores Originais (Aleatórias)', fontsize=14)
ax2.set_xlabel('Posição X')
ax2.set_ylabel('Posição Y')

plt.tight_layout()
plt.show()

print("\n✅ Observe como cores similares ficam próximas no mapa SOM!")
print("   • Vermelhos juntos, azuis juntos, verdes juntos")
print("   • Gradientes suaves entre cores")
print("   • Estrutura auto-organizada!")
