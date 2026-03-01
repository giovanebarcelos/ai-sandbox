# GO0813-PerfilDosClustersInterpretação
# ═══════════════════════════════════════════════════════════════════
# PERFIL DOS CLUSTERS - INTERPRETAÇÃO
# ═══════════════════════════════════════════════════════════════════

print("="*60)
print("PERFIL DETALHADO DOS CLUSTERS")
print("="*60)

feature_names = wine.feature_names

for cluster_id in range(3):
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*60}")

    cluster_wines = df[df['som_cluster'] == cluster_id]

    print(f"Número de vinhos: {len(cluster_wines)}")
    print(f"\nClasses presentes:")
    for class_id in cluster_wines['class'].unique():
        count = (cluster_wines['class'] == class_id).sum()
        pct = count / len(cluster_wines) * 100
        print(f"  {wine.target_names[class_id]}: {count} ({pct:.1f}%)")

    print(f"\nCaracterísticas médias:")
    cluster_means = cluster_wines[feature_names].mean()

    # Comparar com média global
    global_means = df[feature_names].mean()

    # Top 5 features que mais diferem da média
    diff = (cluster_means - global_means).abs().sort_values(ascending=False)

    print(f"\nTop 5 características distintivas:")
    for i, (feature, diff_val) in enumerate(diff.head(5).items(), 1):
        cluster_val = cluster_means[feature]
        global_val = global_means[feature]
        direction = "⬆ maior" if cluster_val > global_val else "⬇ menor"
        print(f"  {i}. {feature}: {cluster_val:.2f} ({direction} que média {global_val:.2f})")

# ───────────────────────────────────────────────────────────────────
# RADAR CHART COMPARATIVO
# ───────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from math import pi

# Selecionar features para radar (normalizar 0-1)
features_for_radar = ['alcohol', 'flavanoids', 'color_intensity', 'proline', 'malic_acid']
n_features = len(features_for_radar)

angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
angles += angles[:1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

for cluster_id in range(3):
    ax = axes[cluster_id]
    cluster_wines = df[df['som_cluster'] == cluster_id]

    # Normalizar features 0-1
    values = []
    for feature in features_for_radar:
        val = cluster_wines[feature].mean()
        min_val = df[feature].min()
        max_val = df[feature].max()
        normalized = (val - min_val) / (max_val - min_val)
        values.append(normalized)

    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features_for_radar, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'Cluster {cluster_id} ({len(cluster_wines)} vinhos)', size=14, pad=20)
    ax.grid(True)

plt.suptitle('Perfil dos Clusters - Radar Chart', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

print("\n✅ Análise de clusters completa!")
