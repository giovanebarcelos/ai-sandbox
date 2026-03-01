# GO0627-Problema4CrossvalidationComResultadosInconsistente
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# Verificar variabilidade:


if __name__ == "__main__":
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Scores: {scores}")
    print(f"Média: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Se std > 0.2, investigar:

    # 1. Aumentar número de folds:
    scores = cross_val_score(model, X, y, cv=10, scoring='r2')

    # 2. Usar Repeated K-Fold:
    from sklearn.model_selection import RepeatedKFold
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    # 3. Verificar outliers:
    from scipy import stats
    z_scores = np.abs(stats.zscore(y))
    outliers = z_scores > 3
    print(f"Outliers encontrados: {outliers.sum()}")

    # Remover outliers:
    X_clean = X[~outliers]
    y_clean = y[~outliers]
