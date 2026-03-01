# GO0625-Problema2OverfittingEmRegressãoPolinomial
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_regression

# Gerar dados de exemplo
X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*60)
print("SELEÇÃO DE GRAU POLINOMIAL COM VALIDAÇÃO CRUZADA")
print("="*60)

# ❌ ERRADO - grau muito alto sem regularização:
print("\n❌ ERRADO: Grau 15 sem regularização")
poly_bad = PolynomialFeatures(degree=15, include_bias=False)
X_poly_bad = poly_bad.fit_transform(X_train_scaled)
model_bad = LinearRegression()
model_bad.fit(X_poly_bad, y_train)
print(f"   Features criadas: {X_poly_bad.shape[1]}")
print(f"   R² Treino: {model_bad.score(X_poly_bad, y_train):.4f}")
print(f"   ⚠️  Risco de overfitting muito alto!")

# ✅ CORRETO - grau moderado + regularização:
print("\n✅ CORRETO: Buscar melhor grau com CV + Ridge")
degrees = [1, 2, 3, 4, 5]
best_degree = None
best_score = -np.inf

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train_scaled)

    # Usar Ridge com regularização
    model = Ridge(alpha=10.0)

    # Validação cruzada
    scores = cross_val_score(model, X_poly, y_train, cv=5, 
                            scoring='neg_mean_squared_error')
    score = scores.mean()

    print(f"   Grau {degree}: MSE CV = {-score:.2f}")

    if score > best_score:
        best_score = score
        best_degree = degree

    print(f"\n🎯 Melhor grau: {best_degree}")
    print(f"   MSE CV: {-best_score:.2f}")

    # Usar Ridge com regularização:
    model = Ridge(alpha=10.0)

    # Validação cruzada:
    scores = cross_val_score(model, X_poly, y_train, cv=5, 
                            scoring='neg_mean_squared_error')
    score = scores.mean()
    if score > best_score:
        best_score = score
        best_degree = degree

print(f"\n🎯 Melhor grau: {best_degree}")
print(f"   MSE CV: {-best_score:.2f}")
