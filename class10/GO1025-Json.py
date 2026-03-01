# GO1025-Json
import json

# Salvar histórico
with open('history.json', 'w') as f:
    json.dump(history.history, f)

# Carregar histórico
with open('history.json', 'r') as f:
    history_loaded = json.load(f)

# Plotar novamente
plt.plot(history_loaded['accuracy'])
