# GO0521-XgboostSklearn
# Instalar: pip install xgboost
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# 1. CARREGAR DATASET (Dígitos 0-9)
# ═══════════════════════════════════════════════════════

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ═══════════════════════════════════════════════════════
# 2. TREINAR XGBoost (Básico)
# ═══════════════════════════════════════════════════════

xgb_model = xgb.XGBClassifier(
    n_estimators=100,          # Número de árvores
    learning_rate=0.1,         # Taxa de aprendizado
    max_depth=5,               # Profundidade máxima
    min_child_weight=1,        # Peso mínimo das folhas
    subsample=0.8,             # % de samples por árvore
    colsample_bytree=0.8,      # % de features por árvore
    gamma=0,                   # Regularização (pruning)
    reg_alpha=0,               # L1 regularization
    reg_lambda=1,              # L2 regularization
    objective='multi:softmax', # Multiclass
    num_class=10,              # 10 classes (0-9)
    eval_metric='mlogloss',    # Métrica de avaliação
    use_label_encoder=False,
    random_state=42
)

# Treinar com validação
eval_set = [(X_train, y_train), (X_test, y_test)]

xgb_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,  # Parar se não melhorar por 10 rounds
    verbose=False              # Não mostrar cada iteração
)

# ═══════════════════════════════════════════════════════
# 3. AVALIAR
# ═══════════════════════════════════════════════════════

y_pred = xgb_model.predict(X_test)

print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\n" + classification_report(y_test, y_pred))

# Melhor iteração (early stopping)
print(f"\nMelhor iteração: {xgb_model.best_iteration}")
print(f"Melhor score: {xgb_model.best_score:.4f}")

# ═══════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE (3 MÉTODOS)
# ═══════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Método 1: Weight (número de vezes que feature foi usada)
xgb.plot_importance(xgb_model, importance_type='weight', ax=axes[0], max_num_features=10)
axes[0].set_title("Feature Importance (Weight)")

# Método 2: Gain (redução média de loss ao usar feature)
xgb.plot_importance(xgb_model, importance_type='gain', ax=axes[1], max_num_features=10)
axes[1].set_title("Feature Importance (Gain)")

# Método 3: Cover (número médio de samples afetados)
xgb.plot_importance(xgb_model, importance_type='cover', ax=axes[2], max_num_features=10)
axes[2].set_title("Feature Importance (Cover)")

plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════
# 5. EVOLUÇÃO DO TREINAMENTO
# ═══════════════════════════════════════════════════════

results = xgb_model.evals_result()

epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
ax.set_ylabel('Log Loss')
ax.set_xlabel('Boosting Round')
ax.set_title('XGBoost Training Progress')
plt.grid(True, alpha=0.3)
plt.show()

# ═══════════════════════════════════════════════════════
# 6. HYPERPARAMETER TUNING (GridSearch)
# ═══════════════════════════════════════════════════════

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_grid = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=10,
    use_label_encoder=False,
    random_state=42
)

grid_search = GridSearchCV(
    xgb_grid,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\n🔍 Executando GridSearch (pode demorar)...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Melhores hiperparâmetros:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n📊 Melhor acurácia (CV): {grid_search.best_score_:.3f}")

# Testar com melhores parâmetros
best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)
print(f"📊 Acurácia no teste: {accuracy_score(y_test, y_pred_best):.3f}")
