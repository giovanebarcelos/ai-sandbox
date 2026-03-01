# GO2002-5MlflowUiECarregarModelos
# Opção 1: carregar específico run
import mlflow.sklearn


if __name__ == "__main__":
    logged_model = 'runs:/RUN_ID/model'  # Substituir RUN_ID
    loaded_model = mlflow.sklearn.load_model(logged_model)
    predictions = loaded_model.predict(X_test)

    # Opção 2: carregar da produção (Model Registry)
    model_name = "iris_rf_model"
    model_version = 1
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
