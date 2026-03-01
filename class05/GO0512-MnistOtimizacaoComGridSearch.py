# GO0512-MnistOtimizaçãoComGridSearch
# ═══════════════════════════════════════════════════════════════════
# MNIST - OTIMIZAÇÃO COM GRID SEARCH
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import GridSearchCV

print("="*60)
print("OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# GRID SEARCH PARA KNN
# ───────────────────────────────────────────────────────────────────

print("\nOtimizando KNN...")

# Usar subset menor para grid search (é lento)
X_grid = X_tr[:2000]
y_grid = y_train[:2000]

param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    param_grid_knn,
    cv=3,
    scoring='accuracy',
    verbose=1
)

knn_grid.fit(X_grid, y_grid)

print(f"\nMelhores hiperparâmetros:")
for param, value in knn_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"Melhor score (CV): {knn_grid.best_score_:.3f}")

# Testar no conjunto completo
best_knn = knn_grid.best_estimator_
best_knn.fit(X_tr, y_train)
y_pred_best = best_knn.predict(X_te)
acc_best = accuracy_score(y_test, y_pred_best)

print(f"\nAcurácia no conjunto de teste completo: {acc_best:.3f}")

# ───────────────────────────────────────────────────────────────────
# GRID SEARCH PARA DECISION TREE
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Otimizando Decision Tree...")

param_grid_tree = {
    'max_depth': [10, 15, 20, 25, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

tree_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_tree,
    cv=3,
    scoring='accuracy',
    verbose=1
)

tree_grid.fit(X_grid, y_grid)

print(f"\nMelhores hiperparâmetros:")
for param, value in tree_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"Melhor score (CV): {tree_grid.best_score_:.3f}")

# ───────────────────────────────────────────────────────────────────
# COMPARAÇÃO FINAL
# ───────────────────────────────────────────────────────────────────

best_tree = tree_grid.best_estimator_
best_tree.fit(X_tr, y_train)
y_pred_tree_best = best_tree.predict(X_te)
acc_tree_best = accuracy_score(y_test, y_pred_tree_best)

print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)
print(f"{'Modelo':<30} {'Antes':<10} {'Depois'}")
print("-" * 60)
print(f"{'KNN':<30} {acc_knn:.3f}      {acc_best:.3f}")
print(f"{'Decision Tree':<30} {acc_tree:.3f}      {acc_tree_best:.3f}")

print(f"\n✅ Modelos otimizados!")
