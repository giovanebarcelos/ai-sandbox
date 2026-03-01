# GO2016-NãoRequerInstalaçãoUsaApenasBibliotecas
import random

@app.post("/predict")
def predict(features: IrisFeatures):
    # Rotear 90% para modelo A, 10% para modelo B
    if random.random() < 0.9:
        model = load_model("modelA")
        model_version = "A"
    else:
        model = load_model("modelB")
        model_version = "B"

    prediction = model.predict(...)

    # Log qual modelo usou
    mlflow.log_param("model_version", model_version)

    return {"prediction": prediction, "model": model_version}
