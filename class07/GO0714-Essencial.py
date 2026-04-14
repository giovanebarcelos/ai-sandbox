# GO0714-Essencial
# Clustering - Conceitos-Chave da Aula 07
# Slide 32: K-Means, DBSCAN, Hierárquico, Silhouette, Davies-Bouldin
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # Gerar dados de exemplo (segmentação de clientes simulada)
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

    # Normalização é CRUCIAL!
    X_scaled = StandardScaler().fit_transform(X)

    # K-Means: rápido, esférico, precisa especificar K
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    # DBSCAN: formas arbitrárias, detecta outliers
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X_scaled)

    # Hierárquico: dendrograma, não precisa K definido de antemão
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=4)
    labels_agg = agg.fit_predict(X_scaled)

    # Métricas: Silhouette (↑ melhor), Davies-Bouldin (↓ melhor)
    from sklearn.metrics import silhouette_score, davies_bouldin_score

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
    print()
    print("Conceitos-Chave (Slide 32):")
    print("  ✅ K-Means: rápido, esférico, precisa especificar K")
    print("  ✅ DBSCAN: formas arbitrárias, detecta outliers")
    print("  ✅ Hierárquico: dendrograma, não precisa K")
    print("  ✅ Silhouette (↑ melhor) | Davies-Bouldin (↓ melhor)")
    print("  ✅ Normalização é CRUCIAL!")
