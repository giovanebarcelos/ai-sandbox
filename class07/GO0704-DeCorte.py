# GO0704-DeCorte
# CLUSTERING HIERÁRQUICO COM SCIPY E SKLEARN

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

# GERAR DADOS

X, y_true = make_blobs(n_samples=100, centers=4, n_features=2,
                       cluster_std=1.0, random_state=42)

# DENDROGRAMA COM SCIPY

# Calcular linkage matrix
# Métodos: 'single', 'complete', 'average', 'ward'
Z = linkage(X, method='ward', metric='euclidean')

plt.figure(figsize=(14, 6))

# Dendrograma completo
plt.subplot(1, 2, 1)
dendrogram(Z)
plt.title('Dendrograma Completo (Ward Linkage)', fontsize=14)
plt.xlabel('Índice da Amostra')
plt.ylabel('Distância')
plt.axhline(y=20, color='r', linestyle='--', label='Corte sugerido')
plt.legend()

# Dendrograma truncado (últimas 10 mesclagens)
plt.subplot(1, 2, 2)
dendrogram(Z, truncate_mode='lastp', p=10)
plt.title('Dendrograma Truncado (últimas 10 mesclagens)', fontsize=14)
plt.xlabel('Cluster')
plt.ylabel('Distância')

plt.tight_layout()
plt.show()

# FORMAR CLUSTERS - Cortar dendrograma

# Método 1: Por distância
clusters_dist = fcluster(Z, t=20, criterion='distance')

# Método 2: Por número de clusters
clusters_k = fcluster(Z, t=4, criterion='maxclust')

print("="*60)
print("FORMAÇÃO DE CLUSTERS")
print("="*60)
print(f"Cortando em distância 20: {len(np.unique(clusters_dist))} clusters")
print(f"Especificando K=4: {len(np.unique(clusters_k))} clusters")

# USAR SKLEARN (mais simples para previsão)

agg_clustering = AgglomerativeClustering(
    n_clusters=4,
    linkage='ward'  # 'ward', 'complete', 'average', 'single'
)

labels = agg_clustering.fit_predict(X)

# COMPARAR DIFERENTES LINKAGES

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
linkages = ['ward', 'complete', 'average', 'single']

for idx, link in enumerate(linkages):
    ax = axes[idx // 2, idx % 2]

    # Clustering
    agg = AgglomerativeClustering(n_clusters=4, linkage=link)
    labels_link = agg.fit_predict(X)

    # Plot
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_link, 
                        cmap='viridis', s=50, edgecolors='k')
    ax.set_title(f'{link.upper()} Linkage', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

print("\n✅ Ward e Average geralmente produzem melhores resultados!")
