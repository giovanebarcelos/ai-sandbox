# GO0515-Problema3NaiveBayesDandoProbabilidades
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────
# CRIAR DADOS E DEMONSTRAR O PROBLEMA
# ───────────────────────────────────────────────────────────────────

print("="*60)
print("TESTANDO SOLUÇÕES PARA NAIVE BAYES COM PROBABILIDADES ESTRANHAS")
print("="*60)

# Carregar dados Iris
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTamanho dos dados:")
print(f"  Treino: {X_train.shape}")
print(f"  Teste: {X_test.shape}")

# ───────────────────────────────────────────────────────────────────
# PROBLEMA: GAUSSIAN NB COM FEATURES NÃO-GAUSSIANAS
# ───────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("PROBLEMA: GaussianNB com features não-gaussianas")
print("-"*60)

# Treinar modelo básico
model_basic = GaussianNB()
model_basic.fit(X_train, y_train)
y_pred_basic = model_basic.predict(X_test)
y_proba_basic = model_basic.predict_proba(X_test)

acc_basic = accuracy_score(y_test, y_pred_basic)
print(f"Acurácia: {acc_basic:.3f}")

# Mostrar algumas probabilidades
print("\nProbabilidades (primeiras 5 amostras):")
for i in range(5):
    print(f"  Amostra {i+1}: {y_proba_basic[i]}")
    # Verificar se tem probabilidades extremas (0.0 ou 1.0)
    if np.any(y_proba_basic[i] < 0.01) or np.any(y_proba_basic[i] > 0.99):
        print(f"    ⚠️  Probabilidade extrema detectada!")

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR DISTRIBUIÇÃO DAS FEATURES
# ───────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("ANÁLISE: Verificando distribuição das features")
print("-"*60)

# Verificar se alguma feature não é gaussiana
for i in range(X.shape[1]):
    skewness = np.abs(np.mean((X_train[:, i] - np.mean(X_train[:, i]))**3) / 
                      (np.std(X_train[:, i])**3))
    print(f"Feature {i} ({iris.feature_names[i][:15]:<15}): skewness = {skewness:.2f}", end="")
    if skewness > 1.0:
        print(" ⚠️ Não-gaussiana!")
    else:
        print(" ✅ OK")

# ───────────────────────────────────────────────────────────────────
# SOLUÇÃO 1: TRANSFORMAR FEATURES PARA TORNÁ-LAS GAUSSIANAS
# ───────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("✅ SOLUÇÃO 1: Transformar features (Yeo-Johnson)")
print("-"*60)

transformer = PowerTransformer(method='yeo-johnson')
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

model_transformed = GaussianNB()
model_transformed.fit(X_train_transformed, y_train)
y_pred_transformed = model_transformed.predict(X_test_transformed)
y_proba_transformed = model_transformed.predict_proba(X_test_transformed)

acc_transformed = accuracy_score(y_test, y_pred_transformed)
print(f"Acurácia: {acc_transformed:.3f}")

print("\nProbabilidades (primeiras 5 amostras após transformação):")
for i in range(5):
    print(f"  Amostra {i+1}: {y_proba_transformed[i]}")

# ───────────────────────────────────────────────────────────────────
# SOLUÇÃO 2: USAR MULTINOMIALNB (PARA DADOS DE CONTAGEM)
# ───────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("✅ SOLUÇÃO 2: MultinomialNB com Laplace smoothing")
print("-"*60)

# MultinomialNB requer features não-negativas
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multiplicar por 100 para simular contagens
X_train_counts = (X_train_scaled * 100).astype(int)
X_test_counts = (X_test_scaled * 100).astype(int)

model_multinomial = MultinomialNB(alpha=1.0)  # Laplace smoothing
model_multinomial.fit(X_train_counts, y_train)
y_pred_multinomial = model_multinomial.predict(X_test_counts)
y_proba_multinomial = model_multinomial.predict_proba(X_test_counts)

acc_multinomial = accuracy_score(y_test, y_pred_multinomial)
print(f"Acurácia: {acc_multinomial:.3f}")

print("\nProbabilidades (primeiras 5 amostras com MultinomialNB):")
for i in range(5):
    print(f"  Amostra {i+1}: {y_proba_multinomial[i]}")

# ───────────────────────────────────────────────────────────────────
# RESUMO COMPARATIVO
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESUMO COMPARATIVO")
print("="*60)
print(f"{'Método':<40} {'Acurácia':<10}")
print("-"*60)
print(f"{'GaussianNB (básico)':<40} {acc_basic:<10.3f}")
print(f"{'GaussianNB + Transformação':<40} {acc_transformed:<10.3f}")
print(f"{'MultinomialNB + Smoothing':<40} {acc_multinomial:<10.3f}")
print("="*60)

print("\n💡 Dicas:")
print("  - Verifique distribuição das features antes de usar GaussianNB")
print("  - Use transformações (Yeo-Johnson, Box-Cox) para tornar dados gaussianos")
print("  - MultinomialNB é melhor para contagens/texto")
print("  - Laplace smoothing (alpha > 0) evita probabilidades zero")
