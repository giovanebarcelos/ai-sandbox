# GO2018-27OptunaFrameworkModernoDeOtimização
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

# Criar study


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize', 
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.MedianPruner())

    # Otimizar
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Resultados
    print(f"\n🏆 Melhor F1-Score: {study.best_value:.4f}")
    print(f"🎯 Melhores parâmetros: {study.best_params}")

    # Visualizar otimização
    import optuna.visualization as vis

    fig1 = vis.plot_optimization_history(study)
    fig2 = vis.plot_param_importances(study)
    fig3 = vis.plot_parallel_coordinate(study)

    fig1.show()
    fig2.show()
    fig3.show()
