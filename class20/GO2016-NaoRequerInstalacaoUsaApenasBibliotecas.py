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


if __name__ == '__main__':
    import random

    print("=== A/B Testing de Modelos (demonstração conceitual) ===")
    print()
    print("  Este código usa FastAPI (@app.post) e requer:")
    print("    pip install fastapi uvicorn")
    print()
    print("  Demo: simulando a lógica de roteamento 90/10 ...")
    print()

    class DummyModel:
        def __init__(self, name):
            self.name = name
        def predict(self, *args, **kwargs):
            return random.choice(["setosa", "versicolor", "virginica"])

    def load_model(version):
        return DummyModel(version)

    contagem = {"A": 0, "B": 0}
    for _ in range(1000):
        if random.random() < 0.9:
            model_version = "A"
        else:
            model_version = "B"
        contagem[model_version] += 1

    total = sum(contagem.values())
    print(f"  Distribuição de 1000 requisições:")
    for v, c in contagem.items():
        print(f"    Modelo {v}: {c} ({c/total*100:.1f}%)")
    print("  (Esperado: ~90% para A, ~10% para B)")
