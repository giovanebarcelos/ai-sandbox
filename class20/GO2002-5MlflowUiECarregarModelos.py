# GO2002-5MlflowUiECarregarModelos
!pip install mlflow

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("iris_classification")

    # Treinar e logar um modelo para depois carregar
    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy_score(y_test, clf.predict(X_test)))
        mlflow.sklearn.log_model(clf, "model")
        run_id = run.info.run_id

    print(f"Run ID gerado: {run_id}")

    # Opção 1: carregar modelo de um run específico
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.sklearn.load_model(logged_model)
    predictions = loaded_model.predict(X_test)
    print(f"\nPredições carregadas via run_id: {predictions}")

    acc = accuracy_score(y_test, predictions)
    print(f"Acurácia do modelo carregado: {acc:.4f}")

    # Opção 2: carregar da produção (Model Registry)
    # Necessita registro prévio (ver GO2004-7MlflowModelRegistry)
    model_name = "iris_rf_model"
    model_version = 1
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        print(f"\nModelo carregado do registry: {model_name} v{model_version}")
    except Exception as e:
        print(f"\nModelo do registry ainda não registrado ({model_name}): {e}")

    # Gráfico: real vs predito
    iris = load_iris()
    plt.figure(figsize=(8, 5))
    indices = np.arange(len(y_test))
    plt.scatter(indices, y_test, label="Real", marker="o", s=80, facecolors="none", edgecolors="blue")
    plt.scatter(indices, predictions, label="Predito (modelo carregado)", marker="x", color="red")
    plt.yticks(range(len(iris.target_names)), iris.target_names)
    plt.xlabel("Amostra (conjunto de teste)")
    plt.title("Predições do modelo carregado via MLflow vs. valores reais")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ver resultados na UI
# mlflow ui
# Acesse http://localhost:5000
