# GO0110-Exploracao
from sklearn.datasets import load_iris

# Carregar dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

# Explorar
print("🌸 Dataset Iris - Classificação de Espécies")
print(f"\nDimensões: {df.shape}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nEstatísticas:\n{df.describe()}")
print(f"\nDistribuição:\n{df['species_name'].value_counts()}")

# Visualizar
plt.figure(figsize=(10, 6))
for species in df['species_name'].unique():
    subset = df[df['species_name'] == species]
    plt.scatter(subset['petal length (cm)'], 
                subset['petal width (cm)'], 
                label=species, s=50, alpha=0.7)
plt.xlabel('Comprimento Pétala (cm)')
plt.ylabel('Largura Pétala (cm)')
plt.title('Iris: Separação por Espécie')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
