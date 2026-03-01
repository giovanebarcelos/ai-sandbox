# GO0814-MapearNovosVinhosNoSomTreinado
# ═══════════════════════════════════════════════════════════════════
# MAPEAR NOVOS VINHOS NO SOM TREINADO
# ═══════════════════════════════════════════════════════════════════

print("="*60)
print("MAPEAMENTO DE NOVOS VINHOS")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# CRIAR VINHOS SINTÉTICOS PARA TESTE
# ───────────────────────────────────────────────────────────────────

# Vinho 1: Similar a classe 0
new_wine_1 = df[df['class'] == 0].iloc[0][feature_names].values
new_wine_1[0] += 0.5  # Pequena variação

# Vinho 2: Similar a classe 1
new_wine_2 = df[df['class'] == 1].iloc[0][feature_names].values
new_wine_2[1] -= 0.3

# Vinho 3: Híbrido (média de duas classes)
new_wine_3 = (df[df['class'] == 0][feature_names].mean() + 
              df[df['class'] == 1][feature_names].mean()) / 2

new_wines = np.array([new_wine_1, new_wine_2, new_wine_3.values])

# Normalizar usando mesmo scaler
new_wines_scaled = scaler.transform(new_wines)

# ───────────────────────────────────────────────────────────────────
# ENCONTRAR BMU PARA CADA NOVO VINHO
# ───────────────────────────────────────────────────────────────────

new_winners = np.array([som.winner(x) for x in new_wines_scaled])
new_clusters = [cluster_map[w[0], w[1]] for w in new_winners]

print("\nNovos vinhos mapeados:")
for i, (winner, cluster) in enumerate(zip(new_winners, new_clusters), 1):
    print(f"\nVinho {i}:")
    print(f"  Posição no SOM: ({winner[0]}, {winner[1]})")
    print(f"  Cluster atribuído: {cluster}")

    # Comparar com vizinhos
    # Encontrar vinhos no mesmo neurônio
    same_neuron = df[(df['som_x'] == winner[0]) & (df['som_y'] == winner[1])]
    if len(same_neuron) > 0:
        print(f"  Vinhos no mesmo neurônio: {len(same_neuron)}")
        class_dist = same_neuron['class'].value_counts()
        print(f"  Distribuição de classes:")
        for class_id, count in class_dist.items():
            print(f"    {wine.target_names[class_id]}: {count}")

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR
# ───────────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 10))

# Plotar todos os vinhos originais (fundo)
for i, name in enumerate(wine.target_names):
    class_winners = winners[y == i]
    plt.scatter(class_winners[:, 0], class_winners[:, 1],
                c=colors[i], label=f'{name} (treino)',
                alpha=0.5, s=100, edgecolors='black', linewidth=0.5)

# Plotar novos vinhos (destaque)
plt.scatter(new_winners[:, 0], new_winners[:, 1],
            c='yellow', marker='*', s=500,
            edgecolors='black', linewidth=2,
            label='Novos vinhos', zorder=5)

# Adicionar labels
for i, winner in enumerate(new_winners, 1):
    plt.annotate(f'Novo {i}', (winner[0], winner[1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Sobrepor grid de clusters
for i in range(som_x):
    for j in range(som_y):
        plt.text(i, j, str(cluster_map[i, j]),
                ha='center', va='center',
                fontsize=8, color='white', weight='bold',
                bbox=dict(boxstyle='circle', facecolor='black', alpha=0.5))

plt.xlim(-0.5, som_x-0.5)
plt.ylim(-0.5, som_y-0.5)
plt.xlabel('Neurônio X', fontsize=12)
plt.ylabel('Neurônio Y', fontsize=12)
plt.title('Mapeamento de Novos Vinhos no SOM', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n✅ Novos vinhos mapeados com sucesso!")
print("\n💡 SOM permite mapear continuamente novos dados sem retreinar!")
