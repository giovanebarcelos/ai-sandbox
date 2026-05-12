# GO1011-Tensorflow
# Compila o modelo com o otimizador Adam de learning rate explícito, loss para
# classificação multiclasse inteira e métrica de acurácia.
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
