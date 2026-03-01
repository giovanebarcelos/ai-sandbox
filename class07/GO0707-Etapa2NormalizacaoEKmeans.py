# GO0707-Etapa2NormalizaçãoEKmeans
# ═══════════════════════════════════════════════════════════════════
# ETAPA 2: NORMALIZAÇÃO E K-MEANS
# ═══════════════════════════════════════════════════════════════════

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ───────────────────────────────────────────────────────────────────
# PREPARAR FEATURES
# ───────────────────────────────────────────────────────────────────

# Selecionar features para clustering
X = df[['Age', 'Annual_Income', 'Spending_Score']].values

print("="*60)
print("PREPARAÇÃO DOS DADOS")
print("="*60)
print(f"Shape original: {X.shape}")
print(f"\nEscalas antes da normalização:")
print(f"  Age:            {X[:, 0].min():.1f} - {X[:, 0].max():.1f}")
print(f"  Annual_Income:  {X[:, 1].min():.1f} - {X[:, 1].max():.1f}")
print(f"  Spending_Score: {X[:, 2].min():.1f} - {X[:, 2].max():.1f}")

# ───────────────────────────────────────────────────────────────────
# NORMALIZAR (CRUCIAL!)
# ───────────────────────────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nApós normalização (média=0, std=1):")
print(f"  Médias: {X_scaled.mean(axis=0).round(10)}")
print(f"  Std:    {X_scaled.std(axis=0).round(2)}")

# ───────────────────────────────────────────────────────────────────
# MÉTODO DO COTOVELO
# ───────────────────────────────────────────────────────────────────

inercias = []
silhouette_scores_list = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)

    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores_list.append(sil_score)

# Plotar
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inercias, marker='o', linewidth=2, markersize=10)
axes[0].set_xlabel('Número de Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inércia', fontsize=12)
axes[0].set_title('Método do Cotovelo', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=4, color='r', linestyle='--', label='K sugerido = 4')
axes[0].legend()

axes[1].plot(K_range, silhouette_scores_list, marker='o', 
            linewidth=2, markersize=10, color='green')
axes[1].set_xlabel('Número de Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=4, color='r', linestyle='--', label='K sugerido = 4')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ANÁLISE DE K")
print("="*60)
for k, inercia, sil in zip(K_range, inercias, silhouette_scores_list):
    print(f"K={k:2d} | Inércia={inercia:7.2f} | Silhouette={sil:.3f}")

print("\n💡 K=4 parece ideal (cotovelo + melhor silhouette)")

# ───────────────────────────────────────────────────────────────────
# TREINAR MODELO FINAL COM K=4
# ───────────────────────────────────────────────────────────────────

kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print("\n" + "="*60)
print("MODELO FINAL (K=4)")
print("="*60)
print(f"Inércia: {kmeans_final.inertia_:.2f}")
print(f"Iterações: {kmeans_final.n_iter_}")
print(f"\nDistribuição de clientes por cluster:")
print(df['Cluster'].value_counts().sort_index())

print("\n✅ ETAPA 2 COMPLETA!")
