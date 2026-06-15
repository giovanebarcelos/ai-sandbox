# GO2003-6MlflowAutolog
!pip install mlflow

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("iris_classification")
    mlflow.autolog()  # Log automático de parâmetros, métricas e modelo

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)

        # Tudo é logado automaticamente!
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Run ID: {run.info.run_id}")
        print(f"Acurácia (calculada manualmente, fora do autolog): {acc:.4f}")

    # Consultar métricas logadas automaticamente
    run_data = mlflow.get_run(run.info.run_id).data
    print("\nParâmetros logados automaticamente:")
    for k, v in run_data.params.items():
        print(f"  {k}: {v}")

    print("\nMétricas logadas automaticamente:")
    for k, v in run_data.metrics.items():
        print(f"  {k}: {v:.4f}")

    # Gráfico de importância das features (do modelo autologado)
    iris = load_iris()
    importances = clf.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(iris.feature_names, importances, color="seagreen")
    plt.xlabel("Importância")
    plt.title("Importância das features (modelo logado via autolog)")
    plt.tight_layout()
    plt.show()
