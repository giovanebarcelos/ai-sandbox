# GO2023-FlaskPrometheus_Flask_Exporter
# src/predict_instrumented.py
!pip install flask prometheus-flask-exporter

import os
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time
import pickle
import numpy as np

app = Flask(__name__)

# ═══════════════════════════════════════════════════════
# 1. MÉTRICAS CUSTOMIZADAS
# ═══════════════════════════════════════════════════════

# Contador de predições
prediction_counter = Counter(
    'ml_predictions_total',
    'Total de predições realizadas',
    ['model_version', 'prediction_class']
)

# Histograma de latência
prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Latência de predição',
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# Gauge de confiança média
confidence_gauge = Gauge(
    'ml_prediction_confidence',
    'Confiança média das predições'
)

# Contador de erros
error_counter = Counter(
    'ml_prediction_errors_total',
    'Total de erros',
    ['error_type']
)

# Gauge de drift (atualizado periodicamente)
data_drift_gauge = Gauge(
    'ml_data_drift_score',
    'Score de data drift (0-1)'
)

# Gauge de acurácia em produção (calculado periodicamente)
production_accuracy_gauge = Gauge(
    'ml_production_accuracy',
    'Acurácia em produção'
)

# ═══════════════════════════════════════════════════════
# 2. CARREGAR MODELO
# ═══════════════════════════════════════════════════════

if not os.path.exists('models/production.pkl'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    os.makedirs('models', exist_ok=True)
    _X, _y = load_iris(return_X_y=True)
    _clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(_X, _y)
    with open('models/production.pkl', 'wb') as f:
        pickle.dump(_clf, f)

with open('models/production.pkl', 'rb') as f:
    model = pickle.load(f)

MODEL_VERSION = "v1.2.3"

# ═══════════════════════════════════════════════════════
# 3. ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    try:
        # Validar entrada
        data = request.get_json()
        if 'features' not in data:
            error_counter.labels(error_type='invalid_input').inc()
            return jsonify({'error': 'Missing features'}), 400

        features = np.array(data['features']).reshape(1, -1)

        # Predição
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()

        # Atualizar métricas
        prediction_counter.labels(
            model_version=MODEL_VERSION,
            prediction_class=str(prediction)
        ).inc()

        confidence_gauge.set(probability)

        # Latência
        latency = time.time() - start_time
        prediction_latency.observe(latency)

        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'model_version': MODEL_VERSION,
            'latency_ms': latency * 1000
        })

    except Exception as e:
        error_counter.labels(error_type='prediction_error').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_version': MODEL_VERSION
    })

# ═══════════════════════════════════════════════════════
# 4. PROMETHEUS METRICS ENDPOINT
# ═══════════════════════════════════════════════════════

# Automaticamente expõe /metrics
metrics = PrometheusMetrics(app)

# Métricas padrão: latência HTTP, requests, etc.
metrics.info('ml_api_info', 'ML API Info', version=MODEL_VERSION)

if __name__ == '__main__':
    # Em produção: app.run(host='0.0.0.0', port=8080)
    # Para demonstração no Colab, usamos o test_client do Flask
    # (simula requisições HTTP sem precisar abrir uma porta).
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    client = app.test_client()

    print(client.get('/health').get_json())

    iris = load_iris()
    latencias = []
    confiancas = []
    for amostra in iris.data[:20]:
        resp = client.post('/predict', json={"features": amostra.tolist()})
        body = resp.get_json()
        latencias.append(body['latency_ms'])
        confiancas.append(body['probability'])

    print(f"\nÚltima predição: {body}")

    # Verificar entrada inválida
    resp_invalida = client.post('/predict', json={})
    print(f"Resposta para entrada inválida: {resp_invalida.get_json()} (status {resp_invalida.status_code})")

    # Conferir métricas expostas pelo Prometheus
    metrics_text = client.get('/metrics').get_data(as_text=True)
    linhas_relevantes = [l for l in metrics_text.splitlines() if l.startswith('ml_')]
    print("\nMétricas Prometheus expostas (/metrics):")
    for l in linhas_relevantes:
        print(f"  {l}")

    # Gráficos: latência e confiança das predições
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(latencias, marker='o', color='steelblue')
    axes[0].set_title("Latência por predição")
    axes[0].set_xlabel("Requisição")
    axes[0].set_ylabel("Latência (ms)")

    axes[1].plot(confiancas, marker='o', color='seagreen')
    axes[1].set_title("Confiança (probabilidade) por predição")
    axes[1].set_xlabel("Requisição")
    axes[1].set_ylabel("Probabilidade")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()
