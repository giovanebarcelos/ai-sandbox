# GO2001-4MlflowTrackingDeExperimentos
!pip install mlflow

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
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
resultados = []

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

            resultados.append({
                "label": f"n_est={n_estimators}, depth={max_depth}",
                "accuracy": acc,
                "f1_score": f1
            })

            print(f"Run: n_est={n_estimators}, depth={max_depth} -> Acc={acc:.4f}")

# Gráfico comparativo das execuções
labels = [r["label"] for r in resultados]
accs = [r["accuracy"] for r in resultados]
f1s = [r["f1_score"] for r in resultados]

x = range(len(resultados))
plt.figure(figsize=(12, 6))
plt.bar([i - 0.2 for i in x], accs, width=0.4, label="Accuracy")
plt.bar([i + 0.2 for i in x], f1s, width=0.4, label="F1-Score")
plt.xticks(list(x), labels, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Comparação de hiperparâmetros - Random Forest (Iris)")
plt.legend()
plt.tight_layout()
plt.show()

# Ver resultados na UI
# mlflow ui
# Acesse http://localhost:5000
