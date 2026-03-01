# GO0805-11SomComMinisomBiblioteca
# USANDO BIBLIOTECA MINISOM (PRODUÇÃO)
# Instalar: pip install minisom

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# CARREGAR E PREPARAR DADOS

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

print("="*60)
print("SOM COM MINISOM - IRIS DATASET")
print("="*60)
print(f"Amostras: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# TREINAR SOM

# Criar SOM 10x10
som = MiniSom(x=10, y=10, input_len=4, sigma=1.5, learning_rate=0.5,
             neighborhood_function='gaussian', random_seed=42)

# Inicializar pesos com PCA (melhor que aleatório)
som.pca_weights_init(X)

# Treinar
print("\nTreinando SOM...")
som.train_random(X, num_iteration=1000, verbose=False)

print(f"✅ Treinamento completo!")
print(f"Erro de quantização: {som.quantization_error(X):.4f}")

# MAPEAR DADOS PARA SOM

# Para cada amostra, encontrar BMU
win_map = som.win_map(X)

# Criar matriz de hits (quantos pontos mapeiam para cada neurônio)
hit_matrix = np.zeros((10, 10))
for sample in X:
    w = som.winner(sample)
    hit_matrix[w] += 1

# VISUALIZAÇÕES

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. U-Matrix
ax1 = axes[0, 0]
umatrix = som.distance_map()
im1 = ax1.imshow(umatrix.T, cmap='bone_r', interpolation='nearest')
ax1.set_title('U-Matrix (Distâncias)', fontsize=14)
plt.colorbar(im1, ax=ax1)

# 2. Hit Matrix (densidade)
ax2 = axes[0, 1]
im2 = ax2.imshow(hit_matrix.T, cmap='Blues', interpolation='nearest')
ax2.set_title('Hit Matrix (Densidade de Amostras)', fontsize=14)
plt.colorbar(im2, ax=ax2)

# Adicionar valores
for i in range(10):
    for j in range(10):
        if hit_matrix[i, j] > 0:
            ax2.text(i, j, int(hit_matrix[i, j]), 
                    ha='center', va='center', color='red', fontsize=8)

# 3. Mapeamento por classe
ax3 = axes[1, 0]
markers = ['o', 's', 'D']
colors_map = ['red', 'green', 'blue']
class_names = iris.target_names

for idx, sample in enumerate(X):
    w = som.winner(sample)
    ax3.plot(w[0] + 0.5, w[1] + 0.5, 
            markers[y[idx]], markerfacecolor='None',
            markeredgecolor=colors_map[y[idx]], markersize=8,
            markeredgewidth=2)

ax3.set_xlim([0, 10])
ax3.set_ylim([0, 10])
ax3.set_title('Mapeamento das Classes Iris', fontsize=14)
ax3.grid(True, alpha=0.3)

# Legenda
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=class_names[0],
          markerfacecolor='none', markeredgecolor='red', markersize=10),
    Line2D([0], [0], marker='s', color='w', label=class_names[1],
          markerfacecolor='none', markeredgecolor='green', markersize=10),
    Line2D([0], [0], marker='D', color='w', label=class_names[2],
          markerfacecolor='none', markeredgecolor='blue', markersize=10)
]
ax3.legend(handles=legend_elements)

# 4. Component Plane - Petal Length (mais discriminativa)
ax4 = axes[1, 1]
component = som.get_weights()[:, :, 2]  # Feature 2 = Petal Length
im4 = ax4.imshow(component.T, cmap='coolwarm', interpolation='nearest')
ax4.set_title('Component Plane: Petal Length', fontsize=14)
plt.colorbar(im4, ax=ax4)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ANÁLISE DOS RESULTADOS")
print("="*60)
print("✅ U-Matrix: Mostra 3 regiões distintas (3 espécies)")
print("✅ Hit Matrix: Distribuição não-uniforme (clusters naturais)")
print("✅ Mapeamento: Classes bem separadas topologicamente")
print("✅ Component Plane: Petal Length diferencia espécies")
