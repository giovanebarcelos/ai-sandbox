# GO0811-VisualizaçõesDoSom
# ═══════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES DO SOM
# ═══════════════════════════════════════════════════════════════════

from matplotlib import cm

fig = plt.figure(figsize=(18, 12))

# ───────────────────────────────────────────────────────────────────
# 1. U-MATRIX
# ───────────────────────────────────────────────────────────────────

ax1 = plt.subplot(2, 3, 1)
u_matrix = som.distance_map()
im1 = ax1.imshow(u_matrix.T, cmap=cm.bone_r, origin='lower', interpolation='bilinear')
ax1.set_title('U-Matrix\n(Fronteiras entre Clusters)', fontsize=12)
ax1.set_xlabel('Neurônio X')
ax1.set_ylabel('Neurônio Y')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# ───────────────────────────────────────────────────────────────────
# 2. MAPA DE DENSIDADE
# ───────────────────────────────────────────────────────────────────

ax2 = plt.subplot(2, 3, 2)
density = np.zeros((som_x, som_y))
for winner in winners:
    density[winner[0], winner[1]] += 1

im2 = ax2.imshow(density.T, cmap='YlOrRd', origin='lower', interpolation='bilinear')
ax2.set_title(f'Mapa de Densidade\n({len(X_scaled)} vinhos)', fontsize=12)
ax2.set_xlabel('Neurônio X')
ax2.set_ylabel('Neurônio Y')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# ───────────────────────────────────────────────────────────────────
# 3. DISTRIBUIÇÃO DE CLASSES
# ───────────────────────────────────────────────────────────────────

ax3 = plt.subplot(2, 3, 3)
colors = ['red', 'green', 'blue']
for i, name in enumerate(wine.target_names):
    class_winners = winners[y == i]
    ax3.scatter(class_winners[:, 0], class_winners[:, 1],
                c=colors[i], label=name, alpha=0.7, s=100, edgecolors='black')

ax3.set_xlim(-0.5, som_x-0.5)
ax3.set_ylim(-0.5, som_y-0.5)
ax3.set_xlabel('Neurônio X')
ax3.set_ylabel('Neurônio Y')
ax3.set_title('Distribuição de Classes no SOM', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# ───────────────────────────────────────────────────────────────────
# 4-6. COMPONENT PLANES (top 3 features)
# ───────────────────────────────────────────────────────────────────

# Identificar features mais importantes (maior variância explicada)
feature_importance = np.var(X_scaled, axis=0)
top_features_idx = np.argsort(feature_importance)[-3:][::-1]

for plot_idx, feat_idx in enumerate(top_features_idx):
    ax = plt.subplot(2, 3, 4 + plot_idx)

    # Criar component plane
    component_plane = np.zeros((som_x, som_y))
    for i in range(som_x):
        for j in range(som_y):
            component_plane[i, j] = som.get_weights()[i, j, feat_idx]

    im = ax.imshow(component_plane.T, cmap='viridis', origin='lower', interpolation='bilinear')
    ax.set_title(f'{wine.feature_names[feat_idx]}', fontsize=10)
    ax.set_xlabel('Neurônio X')
    ax.set_ylabel('Neurônio Y')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Análise Completa do SOM - Dataset de Vinhos', fontsize=16, y=0.995)
plt.tight_layout()
plt.show()

print("="*60)
print("INTERPRETAÇÃO")
print("="*60)
print("U-Matrix: Áreas escuras = fronteiras, áreas claras = clusters")
print("Densidade: Regiões vermelhas = mais vinhos concentrados")
print("Classes: Observe se classes ficaram separadas no grid")
print("Component Planes: Mostram influência de cada feature")
