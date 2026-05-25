# GO1230-Tensorflow
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D
)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# Max Pooling 2×2: seleciona o MAIOR valor de cada janela 2×2
# Vantagem: preserva features mais fortes (bordas, picos de ativação)
# Mais usado em CNNs de classificação onde “tem a feature” é mais importante que “where”
MaxPooling2D(pool_size=(2, 2), strides=2)

# Average Pooling 2×2: calcula a MÉDIA de cada janela 2×2
# Vantagem: preserva informação de fundo/contexto da região
# Usado quando é importante a intensidade média (ex: regressão, redes de detecção)
AveragePooling2D(pool_size=(2, 2))

# Global Average Pooling (GAP): média de CADA feature map inteiro → vetor 1D
# Ex: tensor (7×7×512) → vetor (512,) — substitui Flatten+Dense(FC) em arquiteturas modernas
# Vantagens: (1) muito menos parâmetros que FC, (2) combate overfitting, (3) agnosto a tamanho de entrada
GlobalAveragePooling2D()

# ─── VISUALIZAÇÃO: EFEITO DO POOLING ───
np.random.seed(42)
feature_map = np.array([
    [1, 3, 2, 4, 1, 0],
    [5, 6, 1, 2, 3, 2],
    [2, 1, 8, 5, 4, 1],
    [3, 4, 2, 6, 1, 3],
    [1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5],
], dtype=float)

# Max pooling 2×2 com stride 2
# Aplicar convolução valid manualmente (stride=1, sem padding)
# Para cada posição (i,j): soma element-wise da janela 3×3 com o kernel
def max_pool2d(arr, size=2):
    h, w = arr.shape
    out = np.zeros((h // size, w // size))
    for i in range(0, h, size):
        for j in range(0, w, size):
            # Seleciona o valor máximo da janela size×size
            out[i // size, j // size] = arr[i:i+size, j:j+size].max()
    return out

def avg_pool2d(arr, size=2):
    h, w = arr.shape
    out = np.zeros((h // size, w // size))
    for i in range(0, h, size):
        for j in range(0, w, size):
            # Calcula a média de todos os valores da janela size×size
            out[i // size, j // size] = arr[i:i+size, j:j+size].mean()
    return out

max_pooled = max_pool2d(feature_map)
avg_pooled = avg_pool2d(feature_map)
global_avg = np.mean(feature_map)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(feature_map, cmap='YlOrRd', vmin=0, vmax=8)
axes[0].set_title('Feature Map Original (6×6)', fontsize=12)
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f'{feature_map[i, j]:.0f}', ha='center', va='center',
                     fontsize=11, fontweight='bold',
                     color='white' if feature_map[i, j] > 4 else 'black')
# Desenhar grades do pooling
for k in range(0, 7, 2):
    axes[0].axhline(k - 0.5, color='blue', lw=2)
    axes[0].axvline(k - 0.5, color='blue', lw=2)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_xlabel('Janelas 2×2 (azul) → saída 3×3', fontsize=9)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(max_pooled, cmap='YlOrRd', vmin=0, vmax=8)
axes[1].set_title('Max Pooling 2×2 (3×3)', fontsize=12)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{max_pooled[i, j]:.0f}', ha='center', va='center',
                     fontsize=13, fontweight='bold',
                     color='white' if max_pooled[i, j] > 4 else 'black')
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(avg_pooled, cmap='YlOrRd', vmin=0, vmax=8)
axes[2].set_title('Average Pooling 2×2 (3×3)', fontsize=12)
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f'{avg_pooled[i, j]:.1f}', ha='center', va='center',
                     fontsize=13, fontweight='bold',
                     color='white' if avg_pooled[i, j] > 4 else 'black')
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].set_xlabel(f'Global Avg Pooling → escalar: {global_avg:.2f}', fontsize=9)
plt.colorbar(im2, ax=axes[2])

plt.suptitle('Comparação: Max Pooling vs Average Pooling', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
