# GO0609-AtividadePráticaPrevisãoDePreçosDe
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: PREVISÃO DE PREÇOS DE IMÓVEIS
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset: Boston Housing (sklearn) ou House Prices (Kaggle)
# Vamos usar California Housing por estar built-in

from sklearn.datasets import fetch_california_housing

print("="*60)
print("PREVISÃO DE PREÇOS DE IMÓVEIS - CALIFORNIA HOUSING")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# CARREGAR DADOS
# ───────────────────────────────────────────────────────────────────

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("\nDataset carregado!")
print(f"Shape: {df.shape}")
print(f"\nFeatures:")
for i, col in enumerate(housing.feature_names):
    print(f"  {col}: {housing.feature_names[i]}")

print(f"\nTarget: MedHouseVal (preço mediano em $100k)")

# ───────────────────────────────────────────────────────────────────
# PRIMEIRAS LINHAS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PRIMEIRAS 5 LINHAS")
print("="*60)
print(df.head())

# ───────────────────────────────────────────────────────────────────
# ESTATÍSTICAS DESCRITIVAS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*60)
print(df.describe())

# ───────────────────────────────────────────────────────────────────
# VERIFICAR DADOS FALTANTES
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DADOS FALTANTES")
print("="*60)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Nenhum dado faltante!")
else:
    print(missing[missing > 0])

# ───────────────────────────────────────────────────────────────────
# DISTRIBUIÇÃO DO TARGET
# ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Preço Mediano (×$100k)')
ax1.set_ylabel('Frequência')
ax1.set_title('Distribuição dos Preços')
ax1.grid(True, alpha=0.3)

ax2.boxplot(df['MedHouseVal'])
ax2.set_ylabel('Preço Mediano (×$100k)')
ax2.set_title('Boxplot dos Preços')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Dados carregados e explorados!")
