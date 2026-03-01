# GO0416-Exercicio5ComparacaoEstrategiasSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
sns.set_style('whitegrid')

print("=" * 60)
print("CARREGANDO BREAST CANCER DATASET")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target

print(f"\nDimensões: {X.shape}")
print(f"Proporção classe 1: {(y == 1).sum() / len(y) * 100:.2f}%")
print(f"Proporção classe 0: {(y == 0).sum() / len(y) * 100:.2f}%")

results = {
    'strategy': [],
    'train_accuracy': [],
    'test_accuracy': [],
    'train_f1': [],
    'test_f1': [],
    'train_class_dist': [],
    'test_class_dist': []
}

# 1. HOLDOUT SIMPLES (80/20)
print("\n" + "=" * 60)
print("ESTRATÉGIA 1: HOLDOUT SIMPLES (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTreino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")
print(f"Proporção classe 1 (treino): {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
print(f"Proporção classe 1 (teste): {(y_test == 1).sum() / len(y_test) * 100:.2f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nAcurácia Treino: {train_acc:.4f}")
print(f"Acurácia Teste: {test_acc:.4f}")

results['strategy'].append('Holdout')
results['train_accuracy'].append(train_acc)
results['test_accuracy'].append(test_acc)
results['train_f1'].append(train_f1)
results['test_f1'].append(test_f1)
results['train_class_dist'].append((y_train == 1).sum() / len(y_train))
results['test_class_dist'].append((y_test == 1).sum() / len(y_test))

# 2. STRATIFIED SPLIT
print("\n" + "=" * 60)
print("ESTRATÉGIA 2: STRATIFIED SPLIT")
print("=" * 60)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

print(f"\nTreino: {X_train.shape[0]} amostras")
print(f"Proporção classe 1 (treino): {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
print(f"Proporção classe 1 (teste): {(y_test == 1).sum() / len(y_test) * 100:.2f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nAcurácia Treino: {train_acc:.4f}")
print(f"Acurácia Teste: {test_acc:.4f}")

results['strategy'].append('Stratified')
results['train_accuracy'].append(train_acc)
results['test_accuracy'].append(test_acc)
results['train_f1'].append(train_f1)
results['test_f1'].append(test_f1)
results['train_class_dist'].append((y_train == 1).sum() / len(y_train))
results['test_class_dist'].append((y_test == 1).sum() / len(y_test))

# 3. TIME-BASED SPLIT
print("\n" + "=" * 60)
print("ESTRATÉGIA 3: TIME-BASED SPLIT (Simulado)")
print("=" * 60)

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTreino: {X_train.shape[0]} (primeiras 80%)")
print(f"Teste: {X_test.shape[0]} (últimas 20%)")
print(f"Proporção classe 1 (treino): {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
print(f"Proporção classe 1 (teste): {(y_test == 1).sum() / len(y_test) * 100:.2f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nAcurácia Treino: {train_acc:.4f}")
print(f"Acurácia Teste: {test_acc:.4f}")

results['strategy'].append('Time-based')
results['train_accuracy'].append(train_acc)
results['test_accuracy'].append(test_acc)
results['train_f1'].append(train_f1)
results['test_f1'].append(test_f1)
results['train_class_dist'].append((y_train == 1).sum() / len(y_train))
results['test_class_dist'].append((y_test == 1).sum() / len(y_test))

# Visualização
df_results = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(df_results))
width = 0.35

ax1 = axes[0]
ax1.bar(x - width/2, df_results['train_accuracy'], width, label='Treino', alpha=0.8)
ax1.bar(x + width/2, df_results['test_accuracy'], width, label='Teste', alpha=0.8)
ax1.set_ylabel('Acurácia')
ax1.set_title('Comparação de Acurácia')
ax1.set_xticks(x)
ax1.set_xticklabels(df_results['strategy'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
ax2.bar(x - width/2, df_results['train_class_dist'] * 100, width, label='Treino', alpha=0.8)
ax2.bar(x + width/2, df_results['test_class_dist'] * 100, width, label='Teste', alpha=0.8)
ax2.set_ylabel('% Classe Positiva')
ax2.set_title('Distribuição de Classes')
ax2.set_xticks(x)
ax2.set_xticklabels(df_results['strategy'])
ax2.axhline(y=((y == 1).sum() / len(y)) * 100, color='red', linestyle='--', label='Original')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('split_strategies_comparison.png', dpi=100)
print("\n✓ Gráfico salvo: split_strategies_comparison.png")

print("\n" + "=" * 60)
print("📊 QUANDO USAR CADA ESTRATÉGIA:")
print("=" * 60)
print("\n1. HOLDOUT: Datasets grandes e balanceados")
print("2. STRATIFIED: Datasets desbalanceados (RECOMENDADO)")
print("3. TIME-BASED: Dados temporais/sequenciais\n")
print("✅ Exercício concluído!")
