# GO1026-TensorflowJson
from tensorflow import keras
import json

# 1. Treinar com checkpoint
checkpoint = ModelCheckpoint('best.keras', save_best_only=True)
history = model.fit(X_train, y_train, callbacks=[checkpoint])

# 2. Salvar histórico
with open('history.json', 'w') as f:
    json.dump(history.history, f)

# 3. Mais tarde... carregar modelo
model = keras.models.load_model('best.keras')

# 4. Usar para predição
predictions = model.predict(new_data)
