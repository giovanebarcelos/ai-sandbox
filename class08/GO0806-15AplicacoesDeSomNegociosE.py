# GO0806-15AplicaçõesDeSomNegóciosE
# CASE: SEGMENTAÇÃO DE CLIENTES COM SOM

import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt

# Gerar dados sintéticos de clientes
np.random.seed(42)
n_customers = 500

data = {
    'idade': np.random.randint(18, 70, n_customers),
    'renda': np.random.normal(50000, 20000, n_customers),
    'gastos_anuais': np.random.normal(20000, 10000, n_customers),
    'frequencia_compra': np.random.randint(1, 50, n_customers),
    'tempo_cliente_anos': np.random.randint(0, 10, n_customers)
}

df = pd.DataFrame(data)

# Normalizar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)

print("="*60)
print("SEGMENTAÇÃO DE CLIENTES COM SOM")
print("="*60)
print(f"Clientes: {len(df)}")
print(f"Features: {df.shape[1]}")

# TREINAR SOM

som_shape = (10, 10)  # Grid 10x10
som = MiniSom(som_shape[0], som_shape[1], X.shape[1], 
              sigma=1.5, learning_rate=0.5,
              neighborhood_function='gaussian')

som.random_weights_init(X)
som.train_random(X, 1000)

print(f"\nSOM treinado: grid {som_shape[0]}×{som_shape[1]}")

# MAPEAR CLIENTES

# Para cada cliente, encontrar BMU
winners = np.array([som.winner(x) for x in X])

# Adicionar ao dataframe
df['som_x'] = winners[:, 0]
df['som_y'] = winners[:, 1]
df['cluster'] = df['som_x'] * som_shape[1] + df['som_y']

# VISUALIZAR MAPA DE DENSIDADE

density_map = np.zeros(som_shape)
for winner in winners:
    density_map[winner[0], winner[1]] += 1

plt.figure(figsize=(12, 10))
plt.pcolormesh(density_map.T, cmap='YlOrRd', edgecolors='gray', linewidth=0.5)
plt.colorbar(label='Número de Clientes')
plt.title('Mapa de Densidade de Clientes', fontsize=16)
plt.xlabel('Neurônio X')
plt.ylabel('Neurônio Y')
plt.tight_layout()
plt.show()

# PERFIL DOS CLUSTERS PRINCIPAIS

# Identificar 5 clusters com mais clientes
top_clusters = df['cluster'].value_counts().head(5).index

print("\n" + "="*60)
print("PERFIL DOS TOP 5 CLUSTERS")
print("="*60)

for cluster_id in top_clusters:
    cluster_data = df[df['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_data)} clientes):")
    print(f"  Idade média: {cluster_data['idade'].mean():.1f} anos")
    print(f"  Renda média: R$ {cluster_data['renda'].mean():,.0f}")
    print(f"  Gastos médios: R$ {cluster_data['gastos_anuais'].mean():,.0f}")
    print(f"  Frequência compra: {cluster_data['frequencia_compra'].mean():.1f} vezes/ano")
    print(f"  Tempo cliente: {cluster_data['tempo_cliente_anos'].mean():.1f} anos")

    # Dar um nome ao segmento
    if cluster_data['renda'].mean() > 60000 and cluster_data['gastos_anuais'].mean() > 25000:
        print(f"  📊 Perfil: VIP / Alta Renda")
    elif cluster_data['frequencia_compra'].mean() > 30:
        print(f"  📊 Perfil: Cliente Frequente")
    elif cluster_data['tempo_cliente_anos'].mean() > 5:
        print(f"  📊 Perfil: Cliente Leal")
    else:
        print(f"  📊 Perfil: Casual")

print("\n✅ Segmentação completa!")
