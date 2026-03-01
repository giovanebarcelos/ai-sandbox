# GO1928-Exercício3HyperparameterTuningComAg
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# Cromossomo: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
# Exemplo: [100, 10, 2, 1]
# Ranges: n_estimators [10, 200], max_depth [2, 20], ...

def fitness_rf(params):
    n_est, max_d, min_split, min_leaf = params
    clf = RandomForestClassifier(
        n_estimators=int(n_est),
        max_depth=int(max_d),
        min_samples_split=int(min_split),
        min_samples_leaf=int(min_leaf),
        random_state=42
    )
    X, y = load_iris(return_X_y=True)
    scores = cross_val_score(clf, X, y, cv=5)
    return scores.mean()  # Maximizar acurácia

# TODO: implementar AG com representação inteira


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    print("=== Exercício: Hyperparameter Tuning com Algoritmo Genético ===")
    print("  Fitness = Acurácia 5-fold CV de RandomForest no Iris dataset")
    print()

    # Avaliar alguns conjuntos de hiperparâmetros manualmente
    candidatos = [
        [100, 10, 2, 1],
        [50,   5, 5, 2],
        [200, 15, 3, 1],
        [10,   3, 2, 1],
    ]

    melhor_acc  = 0
    melhor_params = None
    for params in candidatos:
        acc = fitness_rf(params)
        n_est, max_d, min_split, min_leaf = params
        print(f"  n_est={int(n_est):3d}, depth={int(max_d):2d}, "
              f"split={int(min_split)}, leaf={int(min_leaf)} → acc={acc:.4f}")
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_params = params

    print(f"\n  Melhor acc: {melhor_acc:.4f} com params: {melhor_params}")
    print("  Implemente um AG completo para busca automática!")
