# GO0714-Essencial
# Normalizar
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    X_scaled = StandardScaler().fit_transform(X)

    # K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # DBSCAN
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # Hierárquico
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=4)
    labels = agg.fit_predict(X_scaled)

    # Métricas
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, labels)
