# GO1024-Tensorflow
# Configura o ModelCheckpoint para salvar automaticamente somente o modelo com
# melhor val_accuracy durante o treino e recarregá-lo para predição posterior.
from tensorflow.keras.callbacks import ModelCheckpoint

# Salva automaticamente o melhor modelo
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

history = model.fit(X, y, callbacks=[checkpoint])

# Depois do treino
best_model = keras.models.load_model('best_model.keras')
