# GO2004-7MlflowModelRegistry
# Registrar modelo


if __name__ == "__main__":
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, "iris_rf_model")

    # Transições de estágios
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    # Mover para staging
    client.transition_model_version_stage(
        name="iris_rf_model",
        version=1,
        stage="Staging"
    )

    # Mover para produção
    client.transition_model_version_stage(
        name="iris_rf_model",
        version=1,
        stage="Production"
    )

    # Arquivar versão antiga
    client.transition_model_version_stage(
        name="iris_rf_model",
        version=0,
        stage="Archived"
    )
