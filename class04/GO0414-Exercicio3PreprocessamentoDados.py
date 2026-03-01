# GO0414-Exercicio3PreprocessamentoDados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Carregar dataset Wine Quality
print("Carregando dataset Wine Quality...")
# Criar dados sintéticos para demonstração
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'fixed acidity': np.random.normal(8.5, 1.5, n_samples),
    'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
    'citric acid': np.random.normal(0.3, 0.15, n_samples),
    'residual sugar': np.random.exponential(2, n_samples),
    'chlorides': np.random.normal(0.08, 0.04, n_samples),
    'free sulfur dioxide': np.random.normal(15, 10, n_samples),
    'total sulfur dioxide': np.random.normal(50, 30, n_samples),
    'density': np.random.normal(0.997, 0.002, n_samples),
    'pH': np.random.normal(3.3, 0.15, n_samples),
    'sulphates': np.random.normal(0.65, 0.15, n_samples),
    'alcohol': np.random.normal(10.5, 1.2, n_samples),
    'quality': np.random.choice([3, 4, 5, 6, 7, 8], n_samples, p=[0.05, 0.15, 0.30, 0.35, 0.12, 0.03])
})

# Inserir alguns valores faltantes
for col in ['citric acid', 'pH', 'sulphates']:
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, col] = np.nan

print(f"\nDimensões originais: {df.shape}")
print(f"\nPrimeiras linhas:")
print(df.head())

# 2. Tratar valores faltantes
print("\n" + "=" * 60)
print("TRATAMENTO DE VALORES FALTANTES")
print("=" * 60)

missing_before = df.isnull().sum()
print("\nValores faltantes ANTES:")
print(missing_before[missing_before > 0])

# Preencher com a mediana
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().any():
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"✓ '{col}' preenchido com mediana: {median_value:.3f}")

print(f"\nTotal faltantes DEPOIS: {df.isnull().sum().sum()}")

# 3. Detectar e tratar outliers (IQR)
print("\n" + "=" * 60)
print("DETECÇÃO E TRATAMENTO DE OUTLIERS (IQR)")
print("=" * 60)

df_clean = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('quality')

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    if outliers_count > 0:
        print(f"\n{col}: {outliers_count} outliers detectados")
        # Winsorização (clip)
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

print(f"\nDimensões após tratamento: {df_clean.shape}")

# 4. Codificar variáveis categóricas
print("\n" + "=" * 60)
print("CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS")
print("=" * 60)

df_clean['quality_category'] = pd.cut(df_clean['quality'], 
                                      bins=[0, 4, 6, 10],
                                      labels=['Baixa', 'Média', 'Alta'])

print("\nDistribuição de quality_category:")
print(df_clean['quality_category'].value_counts())

# One-Hot Encoding
df_encoded = pd.get_dummies(df_clean, columns=['quality_category'], prefix='quality')
print(f"\n✓ One-Hot Encoding aplicado")

# 5. Normalizar features numéricas
print("\n" + "=" * 60)
print("NORMALIZAÇÃO DE FEATURES")
print("=" * 60)

features_to_scale = [col for col in numeric_cols if col in df_encoded.columns]
scaler = StandardScaler()
df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

print("\nMédias após normalização (devem ser ~0):")
print(df_encoded[features_to_scale].mean().round(4))

# 6. Feature Engineering
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Razão acidez fixa/cítrica
df_encoded['acidity_ratio'] = df['fixed acidity'] / (df['citric acid'] + 0.001)
print("\n✓ Feature 'acidity_ratio' criada")

# Categoria de álcool
df_encoded['alcohol_category'] = pd.cut(df['alcohol'],
                                       bins=[0, 10, 11.5, 100],
                                       labels=['Baixo', 'Médio', 'Alto'])
print("✓ Feature 'alcohol_category' criada")
print("\nDistribuição:")
print(df_encoded['alcohol_category'].value_counts())

# One-hot para alcohol_category
df_final = pd.get_dummies(df_encoded, columns=['alcohol_category'], prefix='alcohol')

# Features adicionais
df_final['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
df_final['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1)

print("\n✓ Features adicionais:")
print("   - total_acidity")
print("   - sulfur_ratio")

# 7. Salvar dataset processado
print("\n" + "=" * 60)
print("SALVANDO DATASET PROCESSADO")
print("=" * 60)

df_final.to_csv('wine_quality_processed.csv', index=False)
print("\n✓ Dataset salvo: wine_quality_processed.csv")
print(f"   Dimensões finais: {df_final.shape}")
print(f"   Features originais: {len(df.columns)}")
print(f"   Features finais: {len(df_final.columns)}")
print(f"   Novas features: {len(df_final.columns) - len(df.columns)}")
print("\n📊 Dataset pronto para modelagem!")
