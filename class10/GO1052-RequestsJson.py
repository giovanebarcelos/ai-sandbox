# GO1052-RequestsJson
# Demonstra como enviar dados ao TensorFlow Serving via HTTP POST e interpretar
# o JSON de resposta, permitindo usar modelos em produção como microsserviços.
import requests
import json

# Preparar dados
# data = {"instances": X_test[:5].tolist()}

# Fazer request
# url = 'http://localhost:8501/v1/models/my_model:predict'
# response = requests.post(url, json=data)

# Parse resultado
# predictions = response.json()['predictions']

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    import numpy as np

    # Simula a resposta do TF Serving com probabilidades fictícias para 5 exemplos
    np.random.seed(42)
    n_examples = 5
    n_classes  = 10
    raw = np.random.rand(n_examples, n_classes)
    predictions_simulated = raw / raw.sum(axis=1, keepdims=True)
    predicted_classes = np.argmax(predictions_simulated, axis=1)

    simulated_response = {
        'predictions': predictions_simulated.tolist()
    }
    print("Resposta simulada do TF Serving:")
    print(json.dumps(
        {'predictions': [[round(v, 4) for v in p] for p in simulated_response['predictions']]},
        indent=2
    ))

    # Gráfico de barras das probabilidades preditas para os 5 exemplos simulados
    fig, axes = plt.subplots(1, n_examples, figsize=(14, 4))
    for i, ax in enumerate(axes):
        ax.bar(range(n_classes), predictions_simulated[i], color='#3498DB')
        ax.set_title(f'Ex {i} → cls {predicted_classes[i]}', fontsize=9)
        ax.set_xticks(range(n_classes))
        ax.set_xlabel('Classe')
        if i == 0:
            ax.set_ylabel('Probabilidade')
    plt.suptitle('TF Serving — Probabilidades Preditas (Simulado)', fontsize=11)
    plt.tight_layout()
    plt.show()
