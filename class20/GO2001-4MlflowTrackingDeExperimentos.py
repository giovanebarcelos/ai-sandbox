# GO2001-4MlflowTrackingDeExperimentos
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Carregar dados
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar experimento
mlflow.set_experiment("iris_classification")

# Treinar com diferentes hiperparâmetros
for n_estimators in [50, 100, 200]:
    for max_depth in [5, 10, None]:
        # Iniciar run
        with mlflow.start_run():
            # Log hiperparâmetros
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("model_type", "RandomForest")

            # Treinar modelo
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            clf.fit(X_train, y_train)

            # Predições
            y_pred = clf.predict(X_test)

            # Log métricas
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Log modelo
            mlflow.sklearn.log_model(clf, "model")

            print(f"Run: n_est={n_estimators}, depth={max_depth} -> Acc={acc:.4f}")

# Ver resultados na UI
# mlflow ui
# Acesse http://localhost:5000
