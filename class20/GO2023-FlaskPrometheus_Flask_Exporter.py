# GO2023-FlaskPrometheus_Flask_Exporter
# src/predict_instrumented.py
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
    app.run(host='0.0.0.0', port=8080)
