# GO0607-FeatureEngineering
# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler,
                                     LabelEncoder, OneHotEncoder)
from sklearn.compose import ColumnTransformer

# Dataset exemplo
data = {
    'tamanho_m2': [50, 80, 120, 150, 200],
    'quartos': [1, 2, 3, 3, 4],
    'bairro': ['Centro', 'Subúrbio', 'Centro', 'Subúrbio', 'Centro'],
    'idade_anos': [5, 15, 2, 20, 10],
    'preco': [200000, 280000, 450000, 380000, 520000]
}

df = pd.DataFrame(data)

print("="*60)
print("FEATURE ENGINEERING")
print("="*60)
print("\nDataset original:")
print(df)

# ───────────────────────────────────────────────────────────────────
# 1. FEATURES POLINOMIAIS
# ───────────────────────────────────────────────────────────────────

poly = PolynomialFeatures(degree=2, include_bias=False)
X_num = df[['tamanho_m2', 'quartos']].values
X_poly = poly.fit_transform(X_num)

print("\n" + "="*60)
print("1. FEATURES POLINOMIAIS (grau 2)")
print("="*60)
print(f"Features originais: {X_num.shape[1]}")
print(f"Features após poly: {X_poly.shape[1]}")
print(f"\nNomes das features:")
print(poly.get_feature_names_out(['tamanho_m2', 'quartos']))

# ───────────────────────────────────────────────────────────────────
# 2. INTERAÇÕES MANUAIS
# ───────────────────────────────────────────────────────────────────

df['m2_por_quarto'] = df['tamanho_m2'] / df['quartos']
df['m2_vezes_quartos'] = df['tamanho_m2'] * df['quartos']

print("\n" + "="*60)
print("2. FEATURES DE INTERAÇÃO")
print("="*60)
print("m²/quarto:", df['m2_por_quarto'].values)
print("m²×quartos:", df['m2_vezes_quartos'].values)

# ───────────────────────────────────────────────────────────────────
# 3. TRANSFORMAÇÕES NÃO-LINEARES
# ───────────────────────────────────────────────────────────────────

df['log_tamanho'] = np.log1p(df['tamanho_m2'])
df['sqrt_idade'] = np.sqrt(df['idade_anos'])

print("\n" + "="*60)
print("3. TRANSFORMAÇÕES NÃO-LINEARES")
print("="*60)
print("log(tamanho):", df['log_tamanho'].values)
print("√idade:", df['sqrt_idade'].values)

# ───────────────────────────────────────────────────────────────────
# 4. ENCODING DE VARIÁVEIS CATEGÓRICAS
# ───────────────────────────────────────────────────────────────────

# One-hot encoding
bairro_encoded = pd.get_dummies(df['bairro'], prefix='bairro')
df = pd.concat([df, bairro_encoded], axis=1)

print("\n" + "="*60)
print("4. ONE-HOT ENCODING")
print("="*60)
print(bairro_encoded)

# ───────────────────────────────────────────────────────────────────
# 5. BINNING (DISCRETIZAÇÃO)
# ───────────────────────────────────────────────────────────────────

df['categoria_idade'] = pd.cut(df['idade_anos'], 
                                bins=[0, 5, 15, 100],
                                labels=['Novo', 'Médio', 'Antigo'])

print("\n" + "="*60)
print("5. BINNING")
print("="*60)
print("Idade → Categoria:")
for idade, cat in zip(df['idade_anos'], df['categoria_idade']):
    print(f"  {idade} anos → {cat}")

print("\n✅ Feature Engineering completo!")
print(f"Features originais: 4")
print(f"Features finais: {df.shape[1] - 1} (excluindo target)")
