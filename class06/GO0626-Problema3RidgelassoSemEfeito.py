# GO0626-Problema3RidgelassoSemEfeito
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# ❌ Sem normalização:
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ✅ Com normalização:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testar diferentes alphas:
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print(f"Alpha={alpha}: R²={score:.4f}, Coefs não-zero={sum(abs(model.coef_) > 0.01)}")
