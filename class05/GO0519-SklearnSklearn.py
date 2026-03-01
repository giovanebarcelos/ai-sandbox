# GO0519-SklearnSklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# 1. CRIAR DATASET
# ═══════════════════════════════════════════════════════

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ═══════════════════════════════════════════════════════
# 2. TREINAR RANDOM FOREST
# ═══════════════════════════════════════════════════════

rf = RandomForestClassifier(
    n_estimators=100,        # 100 árvores
    max_depth=10,            # Profundidade máxima
    min_samples_split=5,     # Mínimo para split
    min_samples_leaf=2,      # Mínimo por folha
    max_features='sqrt',     # √n features por split
    bootstrap=True,          # Usar bootstrap
    oob_score=True,          # Out-of-bag score
    n_jobs=-1,               # Usar todos os CPUs
    random_state=42
)

rf.fit(X_train, y_train)

# ═══════════════════════════════════════════════════════
# 3. AVALIAR
# ═══════════════════════════════════════════════════════

# Acurácia
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
oob_acc = rf.oob_score_  # Out-of-bag (validação interna)

print(f"Acurácia Treino: {train_acc:.3f}")
print(f"Acurácia Teste:  {test_acc:.3f}")
print(f"Acurácia OOB:    {oob_acc:.3f}")  # Similar a cross-validation!

# Predições
y_pred = rf.predict(X_test)

print("\n" + classification_report(y_test, y_pred))

# ═══════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE (CRUCIAL!)
# ═══════════════════════════════════════════════════════

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(20), importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# Top 5 features
print("\nTop 5 Features Mais Importantes:")
for i in range(5):
    idx = indices[i]
    print(f"  Feature {idx}: {importances[idx]:.4f}")

# ═══════════════════════════════════════════════════════
# 5. ANÁLISE DE ÁRVORES INDIVIDUAIS
# ═══════════════════════════════════════════════════════

# Verificar predições de árvores individuais
sample = X_test[0].reshape(1, -1)
tree_predictions = [tree.predict(sample)[0] for tree in rf.estimators_]

print(f"\nPredições das primeiras 10 árvores:")
print(tree_predictions[:10])
print(f"Votação final: {rf.predict(sample)[0]}")
