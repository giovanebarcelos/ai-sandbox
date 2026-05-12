# GO1052-RequestsJson
# Demonstra como enviar dados ao TensorFlow Serving via HTTP POST e interpretar
# o JSON de resposta, permitindo usar modelos em produção como microsserviços.
import requests
import json

# Preparar dados
data = {"instances": X_test[:5].tolist()}

# Fazer request
url = 'http://localhost:8501/v1/models/my_model:predict'
response = requests.post(url, json=data)

# Parse resultado
predictions = response.json()['predictions']
