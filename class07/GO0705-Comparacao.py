# GO0705-Comparação
# MÉTRICAS DE AVALIAÇÃO DE CLUSTERING

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)

# GERAR DADOS

X, y_true = make_blobs(n_samples=500, centers=4, 
                       cluster_std=1.0, random_state=42)

# TESTAR DIFERENTES K

K_range = range(2, 11)
silhouette_scores = []
davies_bouldin_scores = []
calinski_scores = []
inercias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Calcular métricas
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    silhouette_scores.append(sil)
    davies_bouldin_scores.append(db)
    calinski_scores.append(ch)
    inercias.append(kmeans.inertia_)

    print(f"K={k:2d} | Silhouette={sil:.3f} | "
          f"Davies-Bouldin={db:.3f} | Calinski={ch:.1f}")

# VISUALIZAR MÉTRICAS

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Método do Cotovelo
axes[0, 0].plot(K_range, inercias, marker='o', linewidth=2)
axes[0, 0].set_xlabel('Número de Clusters (K)')
axes[0, 0].set_ylabel('Inércia')
axes[0, 0].set_title('Método do Cotovelo')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=4, color='r', linestyle='--', alpha=0.7)

# 2. Silhouette Score (MAIOR melhor)
axes[0, 1].plot(K_range, silhouette_scores, marker='o', 
               linewidth=2, color='green')
axes[0, 1].set_xlabel('Número de Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score (↑ melhor)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=4, color='r', linestyle='--', alpha=0.7)

# 3. Davies-Bouldin (MENOR melhor)
axes[1, 0].plot(K_range, davies_bouldin_scores, marker='o', 
               linewidth=2, color='orange')
axes[1, 0].set_xlabel('Número de Clusters (K)')
axes[1, 0].set_ylabel('Davies-Bouldin Index')
axes[1, 0].set_title('Davies-Bouldin Index (↓ melhor)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=4, color='r', linestyle='--', alpha=0.7)

# 4. Calinski-Harabasz (MAIOR melhor)
axes[1, 1].plot(K_range, calinski_scores, marker='o', 
               linewidth=2, color='purple')
axes[1, 1].set_xlabel('Número de Clusters (K)')
axes[1, 1].set_ylabel('Calinski-Harabasz Score')
axes[1, 1].set_title('Calinski-Harabasz Score (↑ melhor)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=4, color='r', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("CONCLUSÃO: K=4 é ótimo (todas métricas concordam!)")
print("=" * 60)

# SILHOUETTE PLOT DETALHADO

from matplotlib import cm

def plot_silhouette(X, labels, n_clusters):
    """
    Cria gráfico de silhouette para cada cluster
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Silhouette score geral
    silhouette_avg = silhouette_score(X, labels)

    # Silhouette para cada amostra
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        # Valores de silhouette para cluster i
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Plot (K={n_clusters}, Score={silhouette_avg:.3f})')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')

    # Linha vertical do score médio
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
              label=f'Média: {silhouette_avg:.3f}')
    ax.legend()

    plt.show()

# Testar K=4
kmeans_4 = KMeans(n_clusters=4, random_state=42)
labels_4 = kmeans_4.fit_predict(X)
plot_silhouette(X, labels_4, 4)
