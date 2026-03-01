# GO0702-GráficoDoCotovelo
# MÉTODO DO COTOVELO

inercias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inercias.append(kmeans.inertia_)

# Plotar
plt.figure(figsize=(10, 6))
plt.plot(K_range, inercias, marker='o', linewidth=2, markersize=10)
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inércia (WCSS)', fontsize=12)
plt.title('Método do Cotovelo para Escolher K', fontsize=14)
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.axvline(x=3, color='r', linestyle='--', label='K sugerido = 3')
plt.legend()
plt.show()

print("K ótimo parece ser 3 ou 4 (onde a curva achata)")
