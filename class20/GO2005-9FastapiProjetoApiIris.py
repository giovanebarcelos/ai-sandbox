# GO2005-9FastapiProjetoApiIris
!pip install fastapi

import os
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Carregar modelo (treina e salva um modelo de exemplo caso não exista)
if not os.path.exists('model.pkl'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    _X, _y = load_iris(return_X_y=True)
    _clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(_X, _y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(_clf, f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Classes Iris
class_names = ['setosa', 'versicolor', 'virginica']

# Definir schema de entrada
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Criar app
app = FastAPI(title="Iris Prediction API", version="1.0")

# Endpoint raiz
@app.get("/")
def read_root():
    return {"message": "Iris Prediction API is running!"}

# Endpoint de saúde
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Endpoint de predição
@app.post("/predict")
def predict(features: IrisFeatures):
    # Converter para array
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    # Predição
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]

    # Resposta
    return {
        "prediction": class_names[prediction],
        "prediction_index": int(prediction),
        "probabilities": {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
    }

# Endpoint de batch prediction
@app.post("/predict_batch")
def predict_batch(features_list: list[IrisFeatures]):
    data = np.array([[
        f.sepal_length, f.sepal_width, f.petal_length, f.petal_width
    ] for f in features_list])

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    results = []
    for i, pred in enumerate(predictions):
        results.append({
            "prediction": class_names[pred],
            "probabilities": {
                class_names[j]: float(probabilities[i][j])
                for j in range(len(class_names))
            }
        })

    return {"predictions": results}

# Rodar: uvicorn app:app --reload

# Demo local da API usando TestClient (sem precisar rodar um servidor)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from fastapi.testclient import TestClient

    client = TestClient(app)

    print(client.get("/").json())
    print(client.get("/health").json())

    amostra = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    resultado = client.post("/predict", json=amostra).json()
    print("\nPredição:", resultado)

    # Gráfico de probabilidades por classe
    probs = resultado["probabilities"]
    plt.figure(figsize=(6, 4))
    plt.bar(probs.keys(), probs.values(), color=["#4CAF50", "#2196F3", "#FF9800"])
    plt.ylim(0, 1)
    plt.ylabel("Probabilidade")
    plt.title(f"Predição: {resultado['prediction']}")
    plt.tight_layout()
    plt.show()
