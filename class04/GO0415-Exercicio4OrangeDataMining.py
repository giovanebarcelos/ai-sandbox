# GO0415-Exercicio4OrangeDataMining
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

print("=" * 60)
print("EQUIVALENTE EM PYTHON AO ORANGE")
print("=" * 60)

# 1. FILE widget
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(f"\n📊 Dataset: {df.shape}")

# 2. DATA TABLE widget
print("\n" + "=" * 60)
print("DATA TABLE")
print("=" * 60)
print(df.head(10))
print("\nEstatísticas:")
print(df.describe())

# 3. DISTRIBUTIONS widget
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(iris.feature_names):
    ax = axes[i//2, i%2]
    for species in iris.target_names:
        data = df[df['species'] == species][col]
        ax.hist(data, alpha=0.6, label=species, bins=20, edgecolor='black')
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('DISTRIBUTIONS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('orange_distributions.png', dpi=100)
print("\n✓ orange_distributions.png")

# 4. SCATTER PLOT widget
plt.figure(figsize=(10, 7))
for species_name, species_id in zip(iris.target_names, [0, 1, 2]):
    mask = iris.target == species_id
    plt.scatter(iris.data[mask, 0], iris.data[mask, 1],
               label=species_name, alpha=0.7, s=100, edgecolors='black')
plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Sepal Width (cm)', fontsize=12)
plt.title('SCATTER PLOT', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('orange_scatter.png', dpi=100)
print("✓ orange_scatter.png")

# 5. PCA widget
print("\n" + "=" * 60)
print("PCA")
print("=" * 60)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(iris.data)

print(f"\nVariância explicada:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.2f}%")

plt.figure(figsize=(10, 7))
for species_name, species_id in zip(iris.target_names, [0, 1, 2]):
    mask = iris.target == species_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=species_name, alpha=0.7, s=100, edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('orange_pca.png', dpi=100)
print("✓ orange_pca.png")

# 6. TREE widget
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_pca, iris.target)

plt.figure(figsize=(16, 10))
plot_tree(tree_model, filled=True, feature_names=['PC1', 'PC2'],
         class_names=iris.target_names, rounded=True, fontsize=10)
plt.title('DECISION TREE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('orange_tree.png', dpi=100)
print("✓ orange_tree.png")

# 7. TEST & SCORE widget
print("\n" + "=" * 60)
print("TEST & SCORE (CV k=10)")
print("=" * 60)

scores = cross_val_score(tree_model, X_pca, iris.target, cv=10)
print(f"\nAcurácia média: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
print(f"Desvio padrão: {scores.std():.4f}")

print("\n" + "=" * 60)
print("COMPARAÇÃO")
print("=" * 60)
print("\n📊 Orange vs Python:")
print("   • Orange: Interface visual, rápido para prototipagem")
print("   • Python: Flexível, melhor para produção")
print("\n✅ Exercício concluído!")
