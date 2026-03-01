# GO0403-CompletoClassificaçãoDeFloresIris
# ═══════════════════════════════════════════════════════════════════
# EXEMPLO COMPLETO: Classificação de Flores Iris
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ───────────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ───────────────────────────────────────────────────────────────────

iris = load_iris()
X = iris.data  # Features (4): sepal length, sepal width, petal length, petal width
y = iris.target  # Target (3 classes): setosa, versicolor, virginica

# Criar DataFrame para análise
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

print("="*60)
print("DATASET IRIS")
print("="*60)
print(f"\nShape: {df.shape}")
print(f"Features: {list(iris.feature_names)}")
print(f"Classes: {list(iris.target_names)}")
print(f"\nPrimeiras linhas:")
print(df.head())

# ───────────────────────────────────────────────────────────────────
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ESTATÍSTICAS DESCRITIVAS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("DISTRIBUIÇÃO DE CLASSES")
print("="*60)
print(df['species'].value_counts())
print("✅ Dataset balanceado (50 amostras por classe)")

# Visualização: Pairplot
sns.pairplot(df, hue='species', diag_kind='hist')
plt.suptitle('Análise de Pares - Iris Dataset', y=1.02)
plt.show()

# ───────────────────────────────────────────────────────────────────
# 3. PRÉ-PROCESSAMENTO
# ───────────────────────────────────────────────────────────────────

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n" + "="*60)
print("DIVISÃO DE DADOS")
print("="*60)
print(f"Treino: {len(X_train)} amostras ({len(X_train)/len(X)*100:.0f}%)")
print(f"Teste: {len(X_test)} amostras ({len(X_test)/len(X)*100:.0f}%)")

# Normalização (importante para KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Normalização aplicada (StandardScaler)")
print(f"Média treino: {X_train_scaled.mean(axis=0)}")  # ~0
print(f"Desvio padrão treino: {X_train_scaled.std(axis=0)}")  # ~1

# ───────────────────────────────────────────────────────────────────
# 4. TREINAMENTO DO MODELO
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("TREINAMENTO - K-NEAREST NEIGHBORS")
print("="*60)

# Criar e treinar modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

print("✅ Modelo KNN treinado (k=5)")

# ───────────────────────────────────────────────────────────────────
# 5. AVALIAÇÃO
# ───────────────────────────────────────────────────────────────────

# Predições
y_train_pred = knn.predict(X_train_scaled)
y_test_pred = knn.predict(X_test_scaled)

# Acurácia
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(f"Acurácia Treino: {train_acc*100:.2f}%")
print(f"Acurácia Teste: {test_acc*100:.2f}%")

# Relatório detalhado
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusão - KNN (k=5)')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.show()

print("\n" + "="*60)
print("ANÁLISE DE ERROS")
print("="*60)
erros = np.where(y_test != y_test_pred)[0]
print(f"Total de erros: {len(erros)}/{len(y_test)}")

if len(erros) > 0:
    print("\nAmostras incorretas:")
    for idx in erros:
        print(f"  Amostra {idx}: "
              f"Verdadeiro={iris.target_names[y_test[idx]]}, "
              f"Predito={iris.target_names[y_test_pred[idx]]}")
