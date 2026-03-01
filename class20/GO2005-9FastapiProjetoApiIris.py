# GO2005-9FastapiProjetoApiIris
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Carregar modelo
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
