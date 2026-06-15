# GO2004-7MlflowModelRegistry
!pip install mlflow

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # Model Registry requer um backend com suporte a banco de dados (não o file store puro)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("iris_classification")

    # Treinar e logar um modelo para registrar
    with mlflow.start_run() as run:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
        run_id = run.info.run_id

    print(f"Run ID: {run_id} | Acurácia: {acc:.4f}")

    # Registrar modelo
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, "iris_rf_model")
    print(f"Modelo registrado: {mv.name} (versão {mv.version})")

    # Transições de estágios
    client = MlflowClient()

    # Mover para staging
    client.transition_model_version_stage(
        name="iris_rf_model",
        version=mv.version,
        stage="Staging"
    )
    print(f"Versão {mv.version} movida para: Staging")

    # Mover para produção
    client.transition_model_version_stage(
        name="iris_rf_model",
        version=mv.version,
        stage="Production"
    )
    print(f"Versão {mv.version} movida para: Production")

    # Listar todas as versões registradas e seus estágios
    versions = client.search_model_versions(f"name='iris_rf_model'")
    print("\nVersões registradas:")
    for v in versions:
        print(f"  Versão {v.version} -> estágio: {v.current_stage}")

    # Gráfico: estágio de cada versão registrada
    nomes = [f"v{v.version}" for v in versions]
    estagios = [v.current_stage for v in versions]
    cores = {"Production": "green", "Staging": "orange", "Archived": "gray", "None": "lightgray"}

    plt.figure(figsize=(6, 4))
    plt.bar(nomes, [1] * len(nomes), color=[cores.get(e, "blue") for e in estagios])
    for i, e in enumerate(estagios):
        plt.text(i, 0.5, e, ha="center", va="center", color="white", fontweight="bold")
    plt.title("Estágios das versões do modelo no Registry")
    plt.yticks([])
    plt.tight_layout()
    plt.show()
