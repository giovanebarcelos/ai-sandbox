# GO1046-Treinar
# Implementa a fase de fine-tuning: descongela as últimas camadas da base VGG16
# e retreina com learning rate muito baixo para ajustar finamente os pesos à nova tarefa.
# 1. Primeiro, treinar com feature extraction (acima)
# ... train for 10 epochs ...

# 2. Descongelar últimas camadas do base_model
base_model.trainable = True

# Congelar apenas primeiras camadas
for layer in base_model.layers[:-4]:
    layer.trainable = False

# 3. Recompilar com LR MUITO BAIXO
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Treinar mais épocas
model.fit(X_train, y_train, epochs=20)
