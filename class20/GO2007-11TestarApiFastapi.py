# GO2007-11TestarApiFastapi
# Em produção: testa a API rodando em http://localhost:8000 (uvicorn app:app)
#   url = "http://localhost:8000/predict"
#   response = requests.post(url, json=data)
#
# Para demonstração no Colab (sem servidor externo), recriamos a API
# (GO2005-9FastapiProjetoApiIris) e usamos o TestClient do FastAPI,
# que simula as requisições HTTP localmente.
!pip install fastapi

import pickle
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Treinar modelo de exemplo
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
class_names = ['setosa', 'versicolor', 'virginica']

# Recriar a API (igual a GO2005)
app = FastAPI(title="Iris Prediction API", version="1.0")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[features.sepal_length, features.sepal_width,
                       features.petal_length, features.petal_width]])
    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]
    return {
        "prediction": class_names[prediction],
        "prediction_index": int(prediction),
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(3)}
    }


if __name__ == "__main__":
    client = TestClient(app)

    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=data)
    print(response.json())
    # {'prediction': 'setosa', 'prediction_index': 0, 'probabilities': {...}}

    # Gráfico de probabilidades retornadas pela API
    probs = response.json()["probabilities"]
    plt.figure(figsize=(6, 4))
    plt.bar(probs.keys(), probs.values(), color=["#4CAF50", "#2196F3", "#FF9800"])
    plt.ylim(0, 1)
    plt.ylabel("Probabilidade")
    plt.title(f"Resposta da API /predict -> {response.json()['prediction']}")
    plt.tight_layout()
    plt.show()
