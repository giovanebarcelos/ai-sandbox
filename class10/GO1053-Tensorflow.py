# GO1053-Tensorflow
import tensorflow as tf

# Estratégia para múltiplas GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Criar modelo dentro da estratégia
with strategy.scope():
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Treinar normalmente - Keras gerencia distribuição
model.fit(X_train, y_train, epochs=10, batch_size=128)
