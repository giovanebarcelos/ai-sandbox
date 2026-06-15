# GO2022-PytestRequests
# tests/test_api.py
#
# Em produção: testes apontam para uma API rodando em http://localhost:8080
#   BASE_URL = "http://localhost:8080"
#   response = requests.get(f"{BASE_URL}/health")
#
# Para demonstração no Colab (sem servidor externo), recriamos uma API
# equivalente com FastAPI e usamos o TestClient, que simula as
# requisições HTTP localmente (mesma interface do `requests`).
!pip install fastapi

import time
import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Modelo de exemplo
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
MODEL_VERSION = "v1.0.0"

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: dict):
    features = payload.get("features", [])
    if len(features) != 4:
        return JSONResponse(status_code=400, content={"error": "Esperado 4 features"})

    data = np.array([features])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0].max())
    return {"prediction": prediction, "probability": probability, "model_version": MODEL_VERSION}


if __name__ == "__main__":
    client = TestClient(app)
    BASE_URL = ""  # TestClient já aponta para a app

    def test_health_endpoint():
        """Testar endpoint de saúde"""
        response = client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'

    def test_predict_endpoint():
        """Testar endpoint de predição"""
        payload = {"features": [5.1, 3.5, 1.4, 0.2]}
        response = client.post(f"{BASE_URL}/predict", json=payload)

        assert response.status_code == 200
        result = response.json()
        assert 'prediction' in result
        assert 'probability' in result
        assert 'model_version' in result

    def test_predict_validation():
        """Testar validação de entrada"""
        # Entrada inválida (3 features ao invés de 4)
        payload = {"features": [5.1, 3.5, 1.4]}
        response = client.post(f"{BASE_URL}/predict", json=payload)

        assert response.status_code == 400  # Bad Request
        assert 'error' in response.json()

    latencias = []

    def test_predict_load():
        """Testar carga (100 requisições)"""
        payload = {"features": [5.1, 3.5, 1.4, 0.2]}

        for _ in range(100):
            start = time.time()
            response = client.post(f"{BASE_URL}/predict", json=payload)
            latencias.append((time.time() - start) * 1000)
            assert response.status_code == 200

        avg_time_ms = np.mean(latencias)
        print(f"Tempo médio por requisição: {avg_time_ms:.2f}ms")

        # Threshold: 200ms por requisição
        assert avg_time_ms < 200, f"API muito lenta: {avg_time_ms:.2f}ms"

    tests = [
        ("Health endpoint", test_health_endpoint),
        ("Predict endpoint", test_predict_endpoint),
        ("Validação de entrada", test_predict_validation),
        ("Teste de carga (100 req)", test_predict_load),
    ]

    passed, failed = 0, 0
    for nome, fn in tests:
        try:
            fn()
            print(f"  ✅ PASS — {nome}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL — {nome}: {e}")
            failed += 1

    print(f"\n  Resultado: {passed} passou / {failed} falhou")

    # Gráfico: distribuição de latências do teste de carga
    plt.figure(figsize=(8, 4))
    plt.hist(latencias, bins=20, color="cornflowerblue")
    plt.axvline(np.mean(latencias), color="red", linestyle="--", label=f"Média: {np.mean(latencias):.2f}ms")
    plt.xlabel("Latência (ms)")
    plt.ylabel("Nº de requisições")
    plt.title("Distribuição de latência - teste de carga (/predict)")
    plt.legend()
    plt.tight_layout()
    plt.show()
