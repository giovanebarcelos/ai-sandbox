# GO0809-AtividadePráticaAnáliseDeVinhosCom
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: ANÁLISE DE VINHOS COM SOM
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# ───────────────────────────────────────────────────────────────────
# CARREGAR E EXPLORAR DADOS
# ───────────────────────────────────────────────────────────────────

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['class'] = wine.target

print("="*60)
print("ANÁLISE DE VINHOS COM SOM")
print("="*60)
print(f"\nDataset: {df.shape[0]} vinhos, {df.shape[1]-1} features")
print(f"Classes: {wine.target_names}")

print("\nFeatures:")
for i, feature in enumerate(wine.feature_names):
    print(f"  {i+1:2d}. {feature}")

print("\nDistribuição de classes:")
for i, name in enumerate(wine.target_names):
    count = (df['class'] == i).sum()
    print(f"  {name}: {count} ({count/len(df)*100:.1f}%)")

# ───────────────────────────────────────────────────────────────────
# ESTATÍSTICAS DESCRITIVAS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*60)
print(df.describe())

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR DISTRIBUIÇÕES
# ───────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.ravel()

for i, col in enumerate(wine.feature_names[:13]):
    for class_id, name in enumerate(wine.target_names):
        data = df[df['class'] == class_id][col]
        axes[i].hist(data, alpha=0.6, bins=20, label=name)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

# Remover eixos extras
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Distribuição das Features por Classe de Vinho', fontsize=16)
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# MATRIZ DE CORRELAÇÃO
# ───────────────────────────────────────────────────────────────────

plt.figure(figsize=(14, 12))
corr = df.drop('class', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação - Features do Vinho', fontsize=16)
plt.tight_layout()
plt.show()

print("\n✅ Dados explorados!")
