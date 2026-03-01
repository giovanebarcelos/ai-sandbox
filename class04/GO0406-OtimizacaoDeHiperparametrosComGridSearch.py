# GO0406-OtimizaçãoDeHiperparâmetrosComGridSearch
# ═══════════════════════════════════════════════════════════════════
# OTIMIZAÇÃO DE HIPERPARÂMETROS COM GRID SEARCH
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# ───────────────────────────────────────────────────────────────────
# DEFINIR GRID DE HIPERPARÂMETROS
# ───────────────────────────────────────────────────────────────────

param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 8, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy']
}

print("="*60)
print("GRID SEARCH - OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*60)
print(f"Hiperparâmetros a testar:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
print(f"\nTotal de combinações: {total_combinations}")

# ───────────────────────────────────────────────────────────────────
# EXECUTAR GRID SEARCH
# ───────────────────────────────────────────────────────────────────

model = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # usar todos os cores
    verbose=1
)

print("\nIniciando Grid Search...")
grid_search.fit(X, y)

# ───────────────────────────────────────────────────────────────────
# RESULTADOS
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(f"Melhor score (CV): {grid_search.best_score_:.3f}")
print(f"\nMelhores hiperparâmetros:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Top 5 combinações
import pandas as pd
results_df = pd.DataFrame(grid_search.cv_results_)
top5 = results_df.nsmallest(5, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score']
]
print(f"\nTop 5 configurações:")
print(top5.to_string(index=False))

# Modelo final
best_model = grid_search.best_estimator_
print(f"\n✅ Melhor modelo treinado e pronto para uso!")
