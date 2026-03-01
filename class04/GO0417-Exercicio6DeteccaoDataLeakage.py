# GO0417-Exercicio6DeteccaoDataLeakage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

print("=" * 60)
print("DETECÇÃO DE DATA LEAKAGE")
print("=" * 60)

# 1. Criar dataset COM leakage
n_samples = 1000
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)
X3 = np.random.randn(n_samples)
y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

# FEATURE COM LEAKAGE
X4_leakage = y + np.random.normal(0, 0.1, n_samples)  # ⚠️ LEAKAGE!

df = pd.DataFrame({
    'feature_1': X1,
    'feature_2': X2,
    'feature_3': X3,
    'feature_4_LEAKAGE': X4_leakage,
    'target': y
})

print(f"\nDataset criado: {df.shape}")
print("\nPrimeiras linhas:")
print(df.head())

# Correlação
correlation = df.corr()['target'].sort_values(ascending=False)
print("\n" + "=" * 60)
print("CORRELAÇÃO COM TARGET")
print("=" * 60)
print(correlation)

plt.figure(figsize=(10, 6))
correlation.drop('target').plot(kind='barh', 
    color=['red' if abs(x) > 0.3 else 'green' for x in correlation.drop('target')])
plt.xlabel('Correlação')
plt.title('Correlação Features vs Target (Vermelho = Suspeito)')
plt.tight_layout()
plt.savefig('leakage_correlation.png', dpi=100)
print("\n✓ leakage_correlation.png")

# 2. Treinar COM leakage
print("\n" + "=" * 60)
print("MODELO COM LEAKAGE")
print("=" * 60)

X_with = df.drop('target', axis=1)
y_target = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X_with, y_target, test_size=0.2, random_state=42, stratify=y_target
)

model_leak = RandomForestClassifier(n_estimators=100, random_state=42)
model_leak.fit(X_train, y_train)

y_pred_leak = model_leak.predict(X_test)
acc_leak = accuracy_score(y_test, y_pred_leak)

print(f"\n⚠️  Acurácia SUSPEITA: {acc_leak:.4f} ({acc_leak*100:.2f}%)")

# Feature importance
importance = pd.DataFrame({
    'feature': X_with.columns,
    'importance': model_leak.feature_importances_
}).sort_values('importance', ascending=False)

print("\nImportância das Features:")
print(importance)

plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'],
    color=['red' if 'LEAKAGE' in x else 'steelblue' for x in importance['feature']])
plt.xlabel('Importância')
plt.title('Feature Importance (Vermelho = Leakage)')
plt.tight_layout()
plt.savefig('leakage_importance.png', dpi=100)
print("\n✓ leakage_importance.png")

print(f"\n🚨 LEAKAGE DETECTADO:")
print(f"   '{importance.iloc[0]['feature']}' com {importance.iloc[0]['importance']:.4f} importância")

# 4. Treinar SEM leakage
print("\n" + "=" * 60)
print("MODELO SEM LEAKAGE (Corrigido)")
print("=" * 60)

X_clean = df.drop(['target', 'feature_4_LEAKAGE'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_target, test_size=0.2, random_state=42, stratify=y_target
)

model_clean = RandomForestClassifier(n_estimators=100, random_state=42)
model_clean.fit(X_train, y_train)

y_pred_clean = model_clean.predict(X_test)
acc_clean = accuracy_score(y_test, y_pred_clean)

print(f"\n✓ Acurácia REALISTA: {acc_clean:.4f} ({acc_clean*100:.2f}%)")

# Comparação
print("\n" + "=" * 60)
print("COMPARAÇÃO")
print("=" * 60)

comparison = pd.DataFrame({
    'Modelo': ['COM Leakage', 'SEM Leakage'],
    'Acurácia': [acc_leak, acc_clean],
    'Diferença': [acc_leak - acc_clean, 0]
})

print("\n", comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(comparison['Modelo'], comparison['Acurácia'], 
             color=['red', 'green'], alpha=0.7)
ax.set_ylabel('Acurácia')
ax.set_title('Comparação: COM vs SEM Leakage')
ax.set_ylim([0, 1])
for bar, acc in zip(bars, comparison['Acurácia']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
           f'{acc:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('leakage_comparison.png', dpi=100)
print("\n✓ leakage_comparison.png")

# 6. Cenários comuns
print("\n" + "=" * 60)
print("CENÁRIOS COMUNS DE DATA LEAKAGE")
print("=" * 60)

print("""
1️⃣  NORMALIZAÇÃO ANTES DO SPLIT:
   ❌ Normalizar todo dataset antes de dividir
   ✓  Normalizar apenas treino, depois aplicar no teste

2️⃣  FEATURES DERIVADAS DO TARGET:
   ❌ Usar features com informação futura
   ✓  Garantir features calculadas só com passado

3️⃣  DUPLICATAS NO DATASET:
   ❌ Mesmas amostras no treino e teste
   ✓  Remover duplicatas ANTES do split
""")

print("\n✅ Exercício concluído!")
print(f"\nResultado: Acurácia caiu de {acc_leak:.4f} para {acc_clean:.4f}")
print("   Isso é ESPERADO e CORRETO!")
