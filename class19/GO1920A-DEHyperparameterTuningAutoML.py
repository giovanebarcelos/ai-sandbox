# GO1920A-DEHyperparameterTuningAutoML
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution

def rf_objective(params):
    """Otimizar hiperparâmetros Random Forest com DE"""
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])
    max_features = params[4]

    # Treinar modelo
    X, y = load_breast_cancer(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )

    # Cross-validation (3-fold para velocidade)
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')

    # Retornar NEGATIVO (DE minimiza)
    return -scores.mean()

# Espaço de busca (bounds)
bounds = [
    (50, 500),      # n_estimators
    (3, 30),        # max_depth
    (2, 20),        # min_samples_split
    (1, 10),        # min_samples_leaf
    (0.1, 1.0)      # max_features (fração)
]

# Otimizar com DE
result = differential_evolution(
    rf_objective, bounds, popsize=30, maxiter=50
)

best_params = result.x
best_score = result.fun

print(f"🏆 Melhores hiperparâmetros:")
print(f"  n_estimators: {int(best_params[0])}")
print(f"  max_depth: {int(best_params[1])}")
print(f"  min_samples_split: {int(best_params[2])}")
print(f"  min_samples_leaf: {int(best_params[3])}")
print(f"  max_features: {best_params[4]:.3f}")
print(f"  Acurácia: {-best_score:.4f}")
