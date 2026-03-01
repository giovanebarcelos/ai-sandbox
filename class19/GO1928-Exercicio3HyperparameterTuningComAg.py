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
