# GO0603-ValidaçãoCruzadaParaRegressão
# ═══════════════════════════════════════════════════════════════════
# VALIDAÇÃO CRUZADA PARA REGRESSÃO
# ═══════════════════════════════════════════════════════════════════

from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Carregar dados
housing = fetch_california_housing()
X, y = housing.data, housing.target

print("="*60)
print("VALIDAÇÃO CRUZADA - REGRESSÃO")
print("="*60)

# ───────────────────────────────────────────────────────────────────
# K-FOLD CROSS-VALIDATION
# ───────────────────────────────────────────────────────────────────

model = LinearRegression()

# 5-fold CV
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"\n5-Fold Cross-Validation (R²):")
print(f"Scores por fold: {scores}")
print(f"Média: {scores.mean():.3f}")
print(f"Desvio padrão: {scores.std():.3f}")

# ───────────────────────────────────────────────────────────────────
# MÚLTIPLAS MÉTRICAS
# ───────────────────────────────────────────────────────────────────

scoring = {
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}

cv_results = cross_validate(model, X, y, cv=5, scoring=scoring, 
                             return_train_score=True)

print("\n" + "="*60)
print("MÚLTIPLAS MÉTRICAS")
print("="*60)

for metric in ['r2', 'neg_mse', 'neg_mae']:
    test_scores = cv_results[f'test_{metric}']
    train_scores = cv_results[f'train_{metric}']

    print(f"\n{metric.upper()}:")
    print(f"  Treino: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
    print(f"  Teste:  {test_scores.mean():.3f} ± {test_scores.std():.3f}")

# ───────────────────────────────────────────────────────────────────
# LEAVE-ONE-OUT CV (LOOCV)
# ───────────────────────────────────────────────────────────────────

# Usar subset pequeno para LOOCV (seria muito lento com todos)
from sklearn.model_selection import LeaveOneOut
import warnings

X_small = X[:100]
y_small = y[:100]

loo = LeaveOneOut()

# ⚠️ IMPORTANTE: Não usar R² com LOO pois cada fold tem apenas 1 amostra!
# R² requer pelo menos 2 amostras no conjunto de teste.
# Usar MSE ou MAE que funcionam bem com 1 amostra.

# Suprimir warnings de R² com 1 amostra
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    scores_loo_mse = cross_val_score(model, X_small, y_small, cv=loo, 
                                     scoring='neg_mean_squared_error')

print(f"\n" + "="*60)
print("LEAVE-ONE-OUT CV (100 amostras)")
print("="*60)
print(f"MSE médio: {-scores_loo_mse.mean():.3f}")
print(f"Desvio padrão: {scores_loo_mse.std():.3f}")
print(f"⚠️  Nota: LOO usa MSE, não R², pois cada fold tem apenas 1 amostra")
