# GO1024-Tensorflow
# Configura o ModelCheckpoint para salvar automaticamente somente o modelo com
# melhor val_accuracy durante o treino e recarregá-lo para predição posterior.
from tensorflow.keras.callbacks import ModelCheckpoint

# Salva automaticamente o melhor modelo
# checkpoint = ModelCheckpoint(
#     'best_model.keras',
#     monitor='val_accuracy',
#     save_best_only=True,
#     mode='max'
# )

# history = model.fit(X, y, callbacks=[checkpoint])

# Depois do treino
# best_model = keras.models.load_model('best_model.keras')

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Treina com ModelCheckpoint e carrega o melhor modelo salvo automaticamente
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_model_GO1024.keras',
                                 monitor='val_accuracy',
                                 save_best_only=True, mode='max', verbose=0)

    history = model.fit(X_train, y_train, epochs=10,
                        validation_split=0.1,
                        callbacks=[checkpoint], verbose=0)

    val_acc = history.history['val_accuracy']
    best_epoch = int(np.argmax(val_acc))

    # Gráfico de val_accuracy com ponto destacado na melhor época
    plt.figure(figsize=(8, 4))
    plt.plot(val_acc, label='Val Accuracy', marker='o', markersize=4)
    plt.plot(best_epoch, val_acc[best_epoch], 'r*', markersize=14,
             label=f'Melhor época ({best_epoch}) = {val_acc[best_epoch]:.4f}')
    plt.title('Val Accuracy com ModelCheckpoint')
    plt.xlabel('Época')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
