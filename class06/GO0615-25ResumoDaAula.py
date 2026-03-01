# GO0615-25ResumoDaAula
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Normalizar


if __name__ == "__main__":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge
    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='r2')
    print(f"R² (CV): {scores.mean():.3f} ± {scores.std():.3f}")

    # Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    print(f"Features selecionadas: {np.sum(lasso.coef_ != 0)}")
