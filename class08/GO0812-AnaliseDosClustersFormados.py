# GO0812-AnáliseDosClustersFormados
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DOS CLUSTERS FORMADOS
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────
# AGRUPAR NEURÔNIOS EM CLUSTERS (K-Means no grid)
# ───────────────────────────────────────────────────────────────────

from sklearn.cluster import KMeans

# Pegar pesos do SOM e fazer clustering
weights = som.get_weights().reshape(som_x * som_y, n_features)

# Escolher K=3 (sabemos que há 3 classes de vinho)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(weights)

# Reformatar para grid
cluster_map = cluster_labels.reshape(som_x, som_y)

print("="*60)
print("CLUSTERING DO SOM")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR CLUSTERS
# ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Mapa de clusters
im1 = ax1.imshow(cluster_map.T, cmap='viridis', origin='lower', interpolation='nearest')
ax1.set_title('Clusters no SOM (K-Means)', fontsize=14)
ax1.set_xlabel('Neurônio X')
ax1.set_ylabel('Neurônio Y')
plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2], label='Cluster')

# Distribuição de classes sobre clusters
for i, name in enumerate(wine.target_names):
    class_winners = winners[y == i]
    ax2.scatter(class_winners[:, 0], class_winners[:, 1],
                c=colors[i], label=name, alpha=0.7, s=100, edgecolors='black')

# Sobrepor fronteiras dos clusters
from matplotlib.patches import Rectangle
for i in range(som_x):
    for j in range(som_y):
        rect = Rectangle((i-0.5, j-0.5), 1, 1,
                         linewidth=2, edgecolor='white',
                         facecolor='none', alpha=0.3)
        ax2.add_patch(rect)

ax2.set_xlim(-0.5, som_x-0.5)
ax2.set_ylim(-0.5, som_y-0.5)
ax2.set_xlabel('Neurônio X')
ax2.set_ylabel('Neurônio Y')
ax2.set_title('Classes Reais sobre o SOM', fontsize=14)
ax2.legend()
ax2.grid(False)

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# AVALIAR CONCORDÂNCIA CLUSTER vs CLASSE
# ───────────────────────────────────────────────────────────────────

# Para cada vinho, atribuir cluster baseado no BMU
df['som_cluster'] = [cluster_map[w[0], w[1]] for w in winners]

print("\n" + "="*60)
print("DISTRIBUIÇÃO: CLASSE vs CLUSTER")
print("="*60)

confusion = pd.crosstab(df['class'], df['som_cluster'],
                        rownames=['Classe Real'],
                        colnames=['Cluster SOM'])
print(confusion)

# Pureza dos clusters
print("\n" + "="*60)
print("PUREZA DOS CLUSTERS")
print("="*60)

for cluster_id in range(3):
    cluster_data = df[df['som_cluster'] == cluster_id]
    if len(cluster_data) > 0:
        most_common_class = cluster_data['class'].mode()[0]
        purity = (cluster_data['class'] == most_common_class).sum() / len(cluster_data)
        class_name = wine.target_names[most_common_class]
        print(f"Cluster {cluster_id}: {len(cluster_data)} vinhos")
        print(f"  Classe dominante: {class_name}")
        print(f"  Pureza: {purity*100:.1f}%")

# Acurácia ajustada (melhor mapeamento cluster→classe)
from scipy.optimize import linear_sum_assignment
cost_matrix = -confusion.values
row_ind, col_ind = linear_sum_assignment(cost_matrix)
accuracy = confusion.values[row_ind, col_ind].sum() / len(df)

print(f"\nAcurácia (melhor mapeamento): {accuracy*100:.1f}%")
