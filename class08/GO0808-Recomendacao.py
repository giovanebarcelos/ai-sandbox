# GO0808-Recomendação
# ANÁLISE DE HIPERPARÂMETROS DO SOM

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Dataset exemplo (Iris)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

# 1. IMPACTO DO LEARNING RATE (α)

learning_rates = [0.1, 0.5, 1.0, 2.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for i, lr in enumerate(learning_rates):
    som = MiniSom(10, 10, X.shape[1], sigma=1.5, learning_rate=lr)
    som.random_weights_init(X)
    som.train_random(X, 500)

    # Plotar U-Matrix
    from matplotlib import cm
    u_matrix = som.distance_map()
    axes[i].imshow(u_matrix.T, cmap=cm.bone_r, origin='lower')
    axes[i].set_title(f'Learning Rate = {lr}')
    axes[i].set_xlabel('Neurônio X')
    axes[i].set_ylabel('Neurônio Y')

plt.suptitle('Impacto do Learning Rate', fontsize=16)
plt.tight_layout()
plt.show()

print("="*60)
print("HIPERPARÂMETROS DO SOM")
print("="*60)

print("\n1. LEARNING RATE (α₀):")
print("  Muito baixo (0.1): Aprendizado lento, pode não convergir")
print("  Adequado (0.5): Boa convergência")
print("  Alto (2.0): Instável, oscila muito")
print("  Recomendado: 0.3 - 1.0")

# 2. IMPACTO DO SIGMA (raio de vizinhança)

sigmas = [0.5, 1.0, 2.0, 3.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for i, sigma in enumerate(sigmas):
    som = MiniSom(10, 10, X.shape[1], sigma=sigma, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 500)

    u_matrix = som.distance_map()
    axes[i].imshow(u_matrix.T, cmap=cm.bone_r, origin='lower')
    axes[i].set_title(f'Sigma = {sigma}')
    axes[i].set_xlabel('Neurônio X')
    axes[i].set_ylabel('Neurônio Y')

plt.suptitle('Impacto do Sigma (Raio de Vizinhança)', fontsize=16)
plt.tight_layout()
plt.show()

print("\n2. SIGMA (σ₀ - raio inicial):")
print("  Muito baixo (0.5): Mapas fragmentados")
print("  Baixo (1.0): Clusters definidos")
print("  Médio (2.0): Boa topologia")
print("  Alto (3.0): Clusters muito suaves")
print("  Recomendado: max(grid_size) / 2")

# 3. NÚMERO DE ITERAÇÕES

iterations = [100, 500, 1000, 5000]
qe_over_time = []

for n_iter in iterations:
    som = MiniSom(10, 10, X.shape[1], sigma=1.5, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, n_iter)
    qe = som.quantization_error(X)
    qe_over_time.append(qe)
    print(f"\n{n_iter} iterações: QE = {qe:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(iterations, qe_over_time, 'o-', linewidth=2, markersize=8)
plt.xlabel('Número de Iterações')
plt.ylabel('Quantization Error')
plt.title('Convergência do SOM')
plt.grid(True, alpha=0.3)
plt.show()

print("\n3. NÚMERO DE ITERAÇÕES:")
print("  100-500: Poucos, pode não convergir")
print("  500-1000: Adequado para maioria dos casos")
print("  1000-5000: Bom para grids grandes")
print("  Recomendado: 500 * número_de_neurônios")

# ───────────────────────────────────────────────────────────────────
# RESUMO
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESUMO - VALORES RECOMENDADOS")
print("="*60)
print("Grid size:      5√n (n = número de amostras)")
print("Learning rate:  0.5 - 0.7")
print("Sigma inicial:  max(grid_x, grid_y) / 2")
print("Iterações:      500 × (grid_x × grid_y)")
print("Função decay:   'exponential' ou 'linear'")
print("Topologia:      'hexagonal' (mais natural)")
