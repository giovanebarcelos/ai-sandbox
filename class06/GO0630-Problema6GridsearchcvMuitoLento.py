# GO0630-Problema6GridsearchcvMuitoLento
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import time

def main():
    # Gerar dados de exemplo
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("="*60)
    print("COMPARAÇÃO: GRID SEARCH VS RANDOMIZED SEARCH")
    print("="*60)

    # ❌ LENTO - Grid Search completo
    print("\n❌ GRID SEARCH COMPLETO (LENTO)")
    param_grid = {
        'alpha': np.logspace(-4, 4, 20),  # 20 valores
        'max_iter': [1000, 5000, 10000]    # 3 valores
    }
    # Total: 20 × 3 × 5 folds = 300 fits!
    print(f"   Parâmetros: {len(param_grid['alpha'])} alphas × {len(param_grid['max_iter'])} max_iters")
    print(f"   Total de fits: {len(param_grid['alpha']) * len(param_grid['max_iter']) * 5} (com CV=5)")

    start = time.time()
    grid = GridSearchCV(Ridge(), param_grid, cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    elapsed_grid = time.time() - start

    print(f"   Tempo: {elapsed_grid:.2f}s")
    print(f"   Melhor alpha: {grid.best_params_['alpha']:.4f}")
    print(f"   Melhor score: {grid.best_score_:.4f}")

    # ✅ RÁPIDO - Randomized Search
    print("\n✅ RANDOMIZED SEARCH (RÁPIDO)")
    param_dist = {
        'alpha': np.logspace(-4, 4, 100),  # Distribuição contínua
        'max_iter': [1000, 5000, 10000]
    }
    # Testar apenas 20 combinações aleatórias
    print(f"   Distribuição: {100} valores de alpha possíveis")
    print(f"   Total de fits: 20 × 5 = 100 (n_iter=20, CV=5)")

    start = time.time()
    random_search = RandomizedSearchCV(
        Ridge(), param_dist, 
        n_iter=20,  # Apenas 20 fits × 5 folds = 100 fits
        cv=5, 
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    random_search.fit(X_train, y_train)
    elapsed_random = time.time() - start

    print(f"   Tempo: {elapsed_random:.2f}s")
    print(f"   Melhor alpha: {random_search.best_params_['alpha']:.4f}")
    print(f"   Melhor score: {random_search.best_score_:.4f}")

    # Comparação
    print("\n📊 COMPARAÇÃO:")
    print(f"   Speedup: {elapsed_grid / elapsed_random:.1f}x mais rápido!")
    print(f"   Diferença de score: {abs(grid.best_score_ - random_search.best_score_):.4f}")

    print("\n💡 DICA:")
    print("   • Use GridSearchCV para espaços pequenos de hiperparâmetros")
    print("   • Use RandomizedSearchCV para espaços grandes (mais eficiente)")
    print("   • Use menos folds (cv=3) para testes iniciais")

if __name__ == "__main__":
    main()
