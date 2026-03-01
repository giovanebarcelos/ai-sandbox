# GO0510-MnistTreinandoOs3Algoritmos
# ═══════════════════════════════════════════════════════════════════
# MNIST - TREINANDO OS 3 ALGORITMOS
# ═══════════════════════════════════════════════════════════════════

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import time

print("="*60)
print("TREINANDO MODELOS")
print("="*60)

# Vamos usar a normalização simples (/ 255)
X_tr = X_train_normalized
X_te = X_test_normalized

results = {}

# ───────────────────────────────────────────────────────────────────
# 1. K-NEAREST NEIGHBORS
# ───────────────────────────────────────────────────────────────────

print("\n1. Treinando KNN...")
start = time.time()

knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_tr, y_train)
y_pred_knn = knn.predict(X_te)

elapsed_knn = time.time() - start
acc_knn = accuracy_score(y_test, y_pred_knn)

results['KNN'] = {'accuracy': acc_knn, 'time': elapsed_knn}
print(f"  ✅ Concluído em {elapsed_knn:.2f}s")
print(f"  Acurácia: {acc_knn:.3f}")

# ───────────────────────────────────────────────────────────────────
# 2. DECISION TREE
# ───────────────────────────────────────────────────────────────────

print("\n2. Treinando Decision Tree...")
start = time.time()

tree = DecisionTreeClassifier(max_depth=20, random_state=42)
tree.fit(X_tr, y_train)
y_pred_tree = tree.predict(X_te)

elapsed_tree = time.time() - start
acc_tree = accuracy_score(y_test, y_pred_tree)

results['Decision Tree'] = {'accuracy': acc_tree, 'time': elapsed_tree}
print(f"  ✅ Concluído em {elapsed_tree:.2f}s")
print(f"  Acurácia: {acc_tree:.3f}")

# ───────────────────────────────────────────────────────────────────
# 3. NAIVE BAYES
# ───────────────────────────────────────────────────────────────────

print("\n3. Treinando Naive Bayes...")
start = time.time()

nb = GaussianNB()
nb.fit(X_tr, y_train)
y_pred_nb = nb.predict(X_te)

elapsed_nb = time.time() - start
acc_nb = accuracy_score(y_test, y_pred_nb)

results['Naive Bayes'] = {'accuracy': acc_nb, 'time': elapsed_nb}
print(f"  ✅ Concluído em {elapsed_nb:.2f}s")
print(f"  Acurácia: {acc_nb:.3f}")

# ───────────────────────────────────────────────────────────────────
# COMPARAÇÃO
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("COMPARAÇÃO DOS MODELOS")
print("="*60)
print(f"{'Modelo':<20} {'Acurácia':<12} {'Tempo (s)'}")
print("-" * 60)
for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['accuracy']:<12.3f} {metrics['time']:.2f}")

# Melhor modelo
best_model = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n🏆 Melhor modelo: {best_model} ({results[best_model]['accuracy']:.3f})")
