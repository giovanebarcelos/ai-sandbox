# GO1052-RequestsJson
import requests
import json

# Preparar dados
data = {"instances": X_test[:5].tolist()}

# Fazer request
url = 'http://localhost:8501/v1/models/my_model:predict'
response = requests.post(url, json=data)

# Parse resultado
predictions = response.json()['predictions']
