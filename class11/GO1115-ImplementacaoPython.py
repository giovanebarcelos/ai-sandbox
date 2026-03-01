# GO1115-ImplementacaoPython
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt

# Gerar dados aleatórios
data = np.random.rand(100, 2) * 10

# Aplicar FCM
n_clusters = 3
m = 2  # fuzziness parameter
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T,           # dados transpostos (features × samples)
    n_clusters,       # número de clusters
    m,                # fuzziness
    error=0.005,      # critério de parada
    maxiter=1000      # máximo de iterações
)

# cntr = centróides
# u = matriz de pertinência (c × n)
# fpc = Fuzzy Partition Coefficient (qualidade)

# Classificar cada ponto (maior pertinência)
cluster_membership = np.argmax(u, axis=0)

# Visualizar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Clusters com maior pertinência
for i in range(n_clusters):
    ax1.scatter(data[cluster_membership == i, 0],
                data[cluster_membership == i, 1],
                label=f'Cluster {i+1}', alpha=0.6)
ax1.scatter(cntr[:, 0], cntr[:, 1],
             marker='x', s=200, linewidths=3,
             color='black', label='Centróides')
ax1.set_title('FCM Clustering')
ax1.legend()

# Plot 2: Mapa de pertinência (heatmap)
im = ax2.scatter(data[:, 0], data[:, 1],
                 c=u[0, :], cmap='viridis', alpha=0.8)
ax2.set_title('Membership Grade (Cluster 1)')
plt.colorbar(im, ax=ax2, label='μ')

plt.tight_layout()
plt.show()

print(f'FPC: {fpc:.3f}')
