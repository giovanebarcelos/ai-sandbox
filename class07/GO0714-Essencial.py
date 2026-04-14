# GO0714-Essencial
# Clustering - Conceitos-Chave da Aula 07
# Slide 32: K-Means, DBSCAN, Hierárquico, Silhouette, Davies-Bouldin
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    # Gerar dados de exemplo (segmentação de clientes simulada)
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

    # Normalização é CRUCIAL!
    X_scaled = StandardScaler().fit_transform(X)

    # K-Means: rápido, esférico, precisa especificar K
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    # DBSCAN: formas arbitrárias, detecta outliers
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)

    # Hierárquico: dendrograma, não precisa K definido de antemão
    agg = AgglomerativeClustering(n_clusters=4)
    labels_agg = agg.fit_predict(X_scaled)

    # Métricas: Silhouette (↑ melhor), Davies-Bouldin (↓ melhor)
    sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
    db_kmeans = davies_bouldin_score(X_scaled, labels_kmeans)

    # DBSCAN pode ter ruído (label -1); calcular métrica apenas com clusters válidos
    mask = labels_dbscan != -1
    sil_dbscan = silhouette_score(X_scaled[mask], labels_dbscan[mask]) if mask.sum() > 1 else float("nan")
    db_dbscan = davies_bouldin_score(X_scaled[mask], labels_dbscan[mask]) if mask.sum() > 1 else float("nan")

    sil_agg = silhouette_score(X_scaled, labels_agg)
    db_agg = davies_bouldin_score(X_scaled, labels_agg)

    print("=== Resultados de Clustering - Aula 07 ===")
    print(f"{'Algoritmo':<15} {'Silhouette ↑':>14} {'Davies-Bouldin ↓':>18}")
    print("-" * 50)
    print(f"{'K-Means':<15} {sil_kmeans:>14.4f} {db_kmeans:>18.4f}")
    print(f"{'DBSCAN':<15} {sil_dbscan:>14.4f} {db_dbscan:>18.4f}")
    print(f"{'Hierárquico':<15} {sil_agg:>14.4f} {db_agg:>18.4f}")

    # ── Gráfico 1: Método do Cotovelo (Elbow) ──────────────────────────────────
    inertias = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_
                for k in range(1, 9)]

    # ── Gráfico 2: Curva Silhouette por K ──────────────────────────────────────
    sil_scores = [silhouette_score(X_scaled,
                                   KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled))
                  for k in range(2, 9)]

    # ── Figura com 4 subplots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("GO0714-Essencial — Clustering (Aula 07, Slide 32)", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    # Subplot 1 — K-Means
    ax = axes[0, 0]
    for c in np.unique(labels_kmeans):
        ax.scatter(X_scaled[labels_kmeans == c, 0], X_scaled[labels_kmeans == c, 1],
                   s=30, color=colors[c % 10], label=f"Cluster {c}")
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               marker="X", s=180, color="black", zorder=5, label="Centróides")
    ax.set_title(f"K-Means  (Silhouette={sil_kmeans:.3f})")
    ax.set_xlabel("Feature 1 (normalizada)")
    ax.set_ylabel("Feature 2 (normalizada)")
    ax.legend(fontsize=8)

    # Subplot 2 — DBSCAN
    ax = axes[0, 1]
    unique_labels = np.unique(labels_dbscan)
    for lbl in unique_labels:
        mask_lbl = labels_dbscan == lbl
        if lbl == -1:
            ax.scatter(X_scaled[mask_lbl, 0], X_scaled[mask_lbl, 1],
                       s=20, color="gray", marker="x", label="Ruído")
        else:
            ax.scatter(X_scaled[mask_lbl, 0], X_scaled[mask_lbl, 1],
                       s=30, color=colors[lbl % 10], label=f"Cluster {lbl}")
    ax.set_title(f"DBSCAN  (Silhouette={sil_dbscan:.3f})")
    ax.set_xlabel("Feature 1 (normalizada)")
    ax.set_ylabel("Feature 2 (normalizada)")
    ax.legend(fontsize=8)

    # Subplot 3 — Clustering Hierárquico + Dendrograma (truncado)
    ax = axes[1, 0]
    Z = linkage(X_scaled, method="ward")
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=20, leaf_rotation=90, leaf_font_size=8)
    ax.axhline(y=Z[-3, 2], color="red", linestyle="--", linewidth=1.2, label="Corte (K=4)")
    ax.set_title("Dendrograma Hierárquico (Ward)")
    ax.set_xlabel("Amostras (agrupadas)")
    ax.set_ylabel("Distância")
    ax.legend(fontsize=8)

    # Subplot 4 — Elbow + Silhouette
    ax = axes[1, 1]
    ks = range(1, 9)
    ax2 = ax.twinx()
    ax.plot(ks, inertias, "o-", color="steelblue", label="Inércia (Elbow)")
    ax2.plot(range(2, 9), sil_scores, "s--", color="darkorange", label="Silhouette")
    ax.axvline(x=4, color="red", linestyle=":", linewidth=1.2, label="K=4 escolhido")
    ax.set_title("Método do Cotovelo + Silhouette")
    ax.set_xlabel("Número de clusters K")
    ax.set_ylabel("Inércia", color="steelblue")
    ax2.set_ylabel("Silhouette Score", color="darkorange")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    plt.savefig("class07/GO0714-Essencial.png", dpi=120)
    plt.show()
    print("\nGráfico salvo em class07/GO0714-Essencial.png")
