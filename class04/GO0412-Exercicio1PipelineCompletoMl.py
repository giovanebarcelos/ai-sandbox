# GO0412-Exercicio1PipelineCompletoMl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Criar DataFrame para análise exploratória
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# 2. Análise Exploratória
print("=" * 60)
print("ESTATÍSTICAS DESCRITIVAS")
print("=" * 60)
print(df.describe())
print("\nDistribuição das classes:")
print(df['species'].value_counts())

# Visualizações
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Pairplot simplificado
for i, col in enumerate(iris.feature_names[:4]):
    ax = axes[i//2, i%2]
    for species_name, species_id in zip(iris.target_names, [0, 1, 2]):
        ax.scatter(df[df['species'] == species_name][col], 
                  df[df['species'] == species_name].index,
                  label=species_name, alpha=0.6)
    ax.set_xlabel(col)
    ax.set_ylabel('Amostra')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_exploratory.png', dpi=100, bbox_inches='tight')
print("\n✓ Gráfico salvo: iris_exploratory.png")

# 3. Dividir dados em treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTamanho treino: {X_train.shape[0]} amostras")
print(f"Tamanho teste: {X_test.shape[0]} amostras")

# 4. Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nMédias após normalização (treino):", X_train_scaled.mean(axis=0).round(2))
print("Desvios padrão após normalização (treino):", X_train_scaled.std(axis=0).round(2))

# 5. Treinar modelo de Árvore de Decisão
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

print("\n" + "=" * 60)
print("TREINAMENTO CONCLUÍDO")
print("=" * 60)

# 6. Avaliar o modelo
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nAcurácia no treino: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Acurácia no teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "=" * 60)
print("RELATÓRIO DE CLASSIFICAÇÃO (Teste)")
print("=" * 60)
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# Matriz de confusão
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusão - Teste')
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("\n✓ Matriz de confusão salva: confusion_matrix.png")

# Importância das features
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("IMPORTÂNCIA DAS FEATURES")
print("=" * 60)
print(feature_importance.to_string(index=False))
