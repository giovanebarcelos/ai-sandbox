# GO0708-Etapa3AnáliseDosClusters
# ═══════════════════════════════════════════════════════════════════
# ETAPA 3: ANÁLISE DOS CLUSTERS
# ═══════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────
# PERFIL DE CADA CLUSTER
# ───────────────────────────────────────────────────────────────────

print("="*60)
print("PERFIL DOS CLUSTERS")
print("="*60)

cluster_profile = df.groupby('Cluster')[['Age', 'Annual_Income', 
                                         'Spending_Score']].agg(['mean', 'std'])

print(cluster_profile.round(1))

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR CLUSTERS
# ───────────────────────────────────────────────────────────────────

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 12))

# Plot 1: Renda vs Spending Score
ax1 = plt.subplot(2, 3, 1)
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    ax1.scatter(cluster_data['Annual_Income'], 
               cluster_data['Spending_Score'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax1.set_xlabel('Renda Anual (mil)', fontsize=11)
ax1.set_ylabel('Spending Score', fontsize=11)
ax1.set_title('Clusters: Renda vs Spending', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Idade vs Spending Score
ax2 = plt.subplot(2, 3, 2)
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    ax2.scatter(cluster_data['Age'], 
               cluster_data['Spending_Score'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax2.set_xlabel('Idade', fontsize=11)
ax2.set_ylabel('Spending Score', fontsize=11)
ax2.set_title('Clusters: Idade vs Spending', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Idade vs Renda
ax3 = plt.subplot(2, 3, 3)
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    ax3.scatter(cluster_data['Age'], 
               cluster_data['Annual_Income'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax3.set_xlabel('Idade', fontsize=11)
ax3.set_ylabel('Renda Anual (mil)', fontsize=11)
ax3.set_title('Clusters: Idade vs Renda', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: 3D
ax4 = plt.subplot(2, 3, 4, projection='3d')
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    ax4.scatter(cluster_data['Age'], 
               cluster_data['Annual_Income'],
               cluster_data['Spending_Score'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax4.set_xlabel('Idade')
ax4.set_ylabel('Renda (mil)')
ax4.set_zlabel('Spending')
ax4.set_title('Visualização 3D')
ax4.legend()

# Plot 5: Boxplots
ax5 = plt.subplot(2, 3, 5)
df.boxplot(column='Spending_Score', by='Cluster', ax=ax5)
ax5.set_title('Spending Score por Cluster')
ax5.set_xlabel('Cluster')
plt.suptitle('')  # Remove title padrão

# Plot 6: Tamanho dos clusters
ax6 = plt.subplot(2, 3, 6)
cluster_sizes = df['Cluster'].value_counts().sort_index()
ax6.bar(cluster_sizes.index, cluster_sizes.values, 
       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax6.set_xlabel('Cluster')
ax6.set_ylabel('Número de Clientes')
ax6.set_title('Tamanho dos Clusters')
for i, v in enumerate(cluster_sizes.values):
    ax6.text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# NOMEAR E INTERPRETAR CLUSTERS
# ───────────────────────────────────────────────────────────────────

cluster_names = {
    0: "Jovens Econômicos",
    1: "Profissionais Premium",
    2: "Maduros Conservadores",
    3: "Jovens Gastadores"
}

df['Cluster_Name'] = df['Cluster'].map(cluster_names)

print("\n" + "="*60)
print("INTERPRETAÇÃO DOS SEGMENTOS")
print("="*60)

for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\n{cluster_names[cluster]} (Cluster {cluster}):")
    print(f"  Tamanho: {len(cluster_data)} clientes ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"  Idade média: {cluster_data['Age'].mean():.1f} anos")
    print(f"  Renda média: R$ {cluster_data['Annual_Income'].mean():.1f}k")
    print(f"  Spending Score médio: {cluster_data['Spending_Score'].mean():.1f}")

print("\n✅ ETAPA 3 COMPLETA!")
