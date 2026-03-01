# GO2003-6MlflowAutolog
import mlflow
import mlflow.sklearn


if __name__ == "__main__":
    mlflow.autolog()  # Log automático de parâmetros, métricas e modelo

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # Tudo é logado automaticamente!
