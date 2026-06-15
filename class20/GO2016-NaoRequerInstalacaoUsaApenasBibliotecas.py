# GO2016-NãoRequerInstalaçãoUsaApenasBibliotecas
# Código de referência (FastAPI + MLflow - requer pip install fastapi uvicorn mlflow):
#
# import random
#
# @app.post("/predict")
# def predict(features: IrisFeatures):
#     # Rotear 90% para modelo A, 10% para modelo B
#     if random.random() < 0.9:
#         model = load_model("modelA")
#         model_version = "A"
#     else:
#         model = load_model("modelB")
#         model_version = "B"
#
#     prediction = model.predict(...)
#
#     # Log qual modelo usou
#     mlflow.log_param("model_version", model_version)
#
#     return {"prediction": prediction, "model": model_version}

import random
import matplotlib.pyplot as plt


if __name__ == '__main__':
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

    # Gráfico: distribuição real vs esperada do roteamento A/B
    esperado = {"A": total * 0.9, "B": total * 0.1}

    x = list(contagem.keys())
    plt.figure(figsize=(6, 4))
    plt.bar([f"{v} (real)" for v in x], [contagem[v] for v in x], width=0.4, label="Real", color="steelblue")
    plt.bar([f"{v} (esperado)" for v in x], [esperado[v] for v in x], width=0.4, label="Esperado", color="lightgray")
    plt.ylabel("Nº de requisições")
    plt.title("A/B Testing - distribuição real vs esperada (1000 requisições)")
    plt.legend()
    plt.tight_layout()
    plt.show()
