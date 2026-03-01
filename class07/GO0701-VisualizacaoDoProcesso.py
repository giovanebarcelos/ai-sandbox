# GO0701-VisualizaçãoDoProcesso
# K-MEANS DO ZERO E COM SKLEARN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# IMPLEMENTAÇÃO SIMPLES K-MEANS

class SimpleKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # 1. Inicializar centroides aleatoriamente
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iters):
            # 2. Assignment: atribuir cada ponto ao centroide mais próximo
            self.labels = self._assign_clusters(X)

            # 3. Update: recalcular centroides
            old_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X)

            # 4. Verificar convergência
            if np.allclose(old_centroids, self.centroids):
                print(f"Convergiu na iteração {iteration + 1}")
                break

        return self

    def _assign_clusters(self, X):
        # Calcular distância para cada centroide
        distances = np.array([np.linalg.norm(X - centroid, axis=1) 
                            for centroid in self.centroids])
        # Atribuir ao cluster do centroide mais próximo
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X):
        # Calcular nova posição dos centroides
        centroids = np.array([X[self.labels == k].mean(axis=0) 
                             for k in range(self.k)])
        return centroids

    def predict(self, X):
        return self._assign_clusters(X)


# ───────────────────────────────────────────────────────────────────
# EXEMPLO DE USO
# ───────────────────────────────────────────────────────────────────

# Gerar dados sintéticos com 3 clusters
X, y_true = make_blobs(n_samples=300, centers=3, 
                       cluster_std=0.60, random_state=42)

# Treinar K-Means
kmeans_custom = SimpleKMeans(k=3)
kmeans_custom.fit(X)

# Visualizar resultado
plt.figure(figsize=(12, 5))

# Plot 1: Dados originais
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Dados Originais (sem rótulos)', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot 2: Clusters encontrados
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_custom.labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans_custom.centroids[:, 0], kmeans_custom.centroids[:, 1], 
           marker='X', s=300, c='red', edgecolors='black', linewidths=2,
           label='Centroides')
plt.title('Clusters Descobertos pelo K-Means', fontsize=14)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# USANDO SKLEARN (PRODUÇÃO)
# ───────────────────────────────────────────────────────────────────

kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_sklearn.fit(X)

print("="*60)
print("K-MEANS COM SKLEARN")
print("="*60)
print(f"Número de iterações: {kmeans_sklearn.n_iter_}")
print(f"Inércia (soma distâncias²): {kmeans_sklearn.inertia_:.2f}")
print(f"\nCentroides finais:")
print(kmeans_sklearn.cluster_centers_)
print(f"\nDistribuição de pontos por cluster:")
unique, counts = np.unique(kmeans_sklearn.labels_, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} pontos")
