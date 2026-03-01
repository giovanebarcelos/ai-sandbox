# GO0520-SklearnSklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════
# 1. CARREGAR DATASET (Breast Cancer)
# ═══════════════════════════════════════════════════════

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ═══════════════════════════════════════════════════════
# 2. TREINAR GRADIENT BOOSTING
# ═══════════════════════════════════════════════════════

gb = GradientBoostingClassifier(
    n_estimators=100,          # 100 árvores sequenciais
    learning_rate=0.1,         # Taxa de aprendizado (crucial!)
    max_depth=3,               # Árvores RASAS (stumps)
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,             # Stochastic GB (80% dos dados)
    max_features='sqrt',
    verbose=0,
    random_state=42
)

gb.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════
# 3. AVALIAR
# ═══════════════════════════════════════════════════════

train_acc = gb.score(X_train, y_train)
test_acc = gb.score(X_test, y_test)

print(f"Gradient Boosting:")
print(f"  Treino: {train_acc:.3f}")
print(f"  Teste:  {test_acc:.3f}")

# ═══════════════════════════════════════════════════════
# 4. STAGED PREDICTIONS (Evolução do modelo)
# ═══════════════════════════════════════════════════════

# Ver como performance evolui a cada árvore adicionada
train_scores = []
test_scores = []

for i, train_pred in enumerate(gb.staged_predict(X_train)):
    train_scores.append((train_pred == y_train).mean())

for i, test_pred in enumerate(gb.staged_predict(X_test)):
    test_scores.append((test_pred == y_test).mean())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_scores, label='Train', linewidth=2)
plt.plot(test_scores, label='Test', linewidth=2)
plt.xlabel('Número de Árvores (Boosting Iterations)')
plt.ylabel('Acurácia')
plt.title('Evolução do Gradient Boosting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ═══════════════════════════════════════════════════════
# 5. LEARNING RATE EXPERIMENT
# ═══════════════════════════════════════════════════════

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
results = []

for lr in learning_rates:
    gb_lr = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb_lr.fit(X_train, y_train)
    acc = gb_lr.score(X_test, y_test)
    results.append(acc)
    print(f"LR={lr:.2f}: Acurácia={acc:.3f}")

# Melhor learning rate
best_lr = learning_rates[np.argmax(results)]
print(f"\nMelhor Learning Rate: {best_lr}")
