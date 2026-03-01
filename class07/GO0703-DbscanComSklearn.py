# GO0703-DbscanComSklearn
# ═══════════════════════════════════════════════════════════════════
# DBSCAN COM SKLEARN
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ───────────────────────────────────────────────────────────────────
# GERAR DADOS NÃO-LINEARES (formas irregulares)
# ───────────────────────────────────────────────────────────────────

# Dataset 1: Duas "luas"


if __name__ == "__main__":
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # Dataset 2: Círculos concêntricos
    X_circles, _ = make_circles(n_samples=300, noise=0.05, 
                                factor=0.5, random_state=42)

    # ───────────────────────────────────────────────────────────────────
    # APLICAR DBSCAN
    # ───────────────────────────────────────────────────────────────────

    def aplicar_dbscan(X, eps, min_samples, titulo):
        """
        Aplica DBSCAN e visualiza resultados
        """
        # Normalizar dados
        X_scaled = StandardScaler().fit_transform(X)

        # DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # Número de clusters (ignorando ruído)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Plotar
        plt.figure(figsize=(10, 6))

        # Pontos dos clusters
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Ruído em preto
                col = 'black'
                marker = 'x'
                label = 'Ruído'
            else:
                marker = 'o'
                label = f'Cluster {k}'

            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker,
                       s=50, alpha=0.7, label=label, edgecolors='k')

        plt.title(f'{titulo}\n{n_clusters} clusters, {n_noise} outliers', 
                 fontsize=14)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return n_clusters, n_noise

    # ───────────────────────────────────────────────────────────────────
    # TESTAR DIFERENTES PARÂMETROS
    # ───────────────────────────────────────────────────────────────────

    print("="*60)
    print("DBSCAN - MOONS")
    print("="*60)

    # Bons parâmetros
    n_c, n_n = aplicar_dbscan(X_moons, eps=0.3, min_samples=5, 
                              titulo='DBSCAN em Moons (eps=0.3)')

    # ε muito grande → poucos clusters
    aplicar_dbscan(X_moons, eps=0.8, min_samples=5, 
                  titulo='DBSCAN em Moons (eps=0.8 - muito grande!)')

    # ε muito pequeno → muitos outliers
    aplicar_dbscan(X_moons, eps=0.1, min_samples=5, 
                  titulo='DBSCAN em Moons (eps=0.1 - muito pequeno!)')

    print("\n" + "="*60)
    print("DBSCAN - CIRCLES")
    print("="*60)

    aplicar_dbscan(X_circles, eps=0.3, min_samples=5, 
                  titulo='DBSCAN em Circles (eps=0.3)')

    # ───────────────────────────────────────────────────────────────────
    # COMPARAR K-MEANS vs DBSCAN
    # ───────────────────────────────────────────────────────────────────

    from sklearn.cluster import KMeans

    X_scaled = StandardScaler().fit_transform(X_moons)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # K-Means (FALHA!)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_km = kmeans.fit_predict(X_scaled)
    axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_km, 
                   cmap='viridis', s=50, edgecolors='k')
    axes[0].set_title('K-Means (falha em formas não-convexas)', fontsize=13)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # DBSCAN (SUCESSO!)
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels_db = dbscan.fit_predict(X_scaled)
    axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db, 
                   cmap='viridis', s=50, edgecolors='k')
    axes[1].set_title('DBSCAN (detecta formas corretamente!)', fontsize=13)
    axes[1].set_xlabel('Feature 1')

    plt.tight_layout()
    plt.show()

    print("\n✅ DBSCAN identifica corretamente as duas 'luas'!")
    print("❌ K-Means falha porque assume clusters esféricos")
