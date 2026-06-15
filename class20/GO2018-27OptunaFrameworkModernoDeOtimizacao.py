# GO2018-27OptunaFrameworkModernoDeOtimização
!pip install optuna

import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    """Função objetivo para Optuna"""
    # Definir espaço de busca
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }

    # Treinar modelo
    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    # Cross-validation
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro').mean()

    return score


if __name__ == "__main__":
    # Criar study (n_trials reduzido para demonstração rápida)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())

    # Otimizar
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # Resultados
    print(f"\n🏆 Melhor F1-Score: {study.best_value:.4f}")
    print(f"🎯 Melhores parâmetros: {study.best_params}")

    # Visualizar otimização (gráficos em matplotlib)
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
    )

    plot_optimization_history(study)
    plt.tight_layout()
    plt.show()

    plot_param_importances(study)
    plt.tight_layout()
    plt.show()

    plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.show()
