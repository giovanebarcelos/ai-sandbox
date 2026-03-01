# GO2022-PytestRequests
# tests/test_api.py
import pytest
import requests
import json


if __name__ == "__main__":
    BASE_URL = "http://localhost:8080"

    def test_health_endpoint():
        """Testar endpoint de saúde"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'

    def test_predict_endpoint():
        """Testar endpoint de predição"""
        payload = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload
        )

        assert response.status_code == 200
        result = response.json()

        assert 'prediction' in result
        assert 'probability' in result
        assert 'model_version' in result

    def test_predict_validation():
        """Testar validação de entrada"""
        # Entrada inválida (3 features ao invés de 4)
        payload = {
            "features": [5.1, 3.5, 1.4]
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload
        )

        assert response.status_code == 400  # Bad Request
        assert 'error' in response.json()

    def test_predict_load():
        """Testar carga (100 requisições)"""
        import time

        payload = {"features": [5.1, 3.5, 1.4, 0.2]}

        start = time.time()
        for _ in range(100):
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            assert response.status_code == 200
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 100) * 1000
        print(f"Tempo médio por requisição: {avg_time_ms:.2f}ms")

        # Threshold: 200ms por requisição
        assert avg_time_ms < 200, f"API muito lenta: {avg_time_ms:.2f}ms"
