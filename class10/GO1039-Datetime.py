# GO1039-Datetime
# Salvar com versão
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'model_{timestamp}.keras')

# Salvar hiperparâmetros
import json
config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 50,
    'dropout': 0.3
}
with open(f'config_{timestamp}.json', 'w') as f:
    json.dump(config, f)
