# GO1031-Inputs
# Demonstra modelo com múltiplas saídas: a mesma representação interna alimenta duas
# cabeças de classificação (dígito e paridade), cada uma com sua própria loss e peso.
# inputs = Input(shape=(784,))
# x = Dense(128, 'relu')(inputs)
# x = Dense(64, 'relu')(x)
#
# # Saída 1: Classificação de dígitos
# output_digit = Dense(10, 'softmax', name='digit')(x)
#
# # Saída 2: Par ou ímpar
# output_parity = Dense(2, 'softmax', name='parity')(x)
#
# model = Model(inputs=inputs, outputs=[output_digit, output_parity])
#
# # Compilar com múltiplas losses
# model.compile(
#     optimizer='adam',
#     loss={
#         'digit': 'sparse_categorical_crossentropy',
#         'parity': 'sparse_categorical_crossentropy'
#     },
#     loss_weights={'digit': 1.0, 'parity': 0.5},
#     metrics=['accuracy']
# )
#
# # Treinar com dois targets
# model.fit(X_train, {'digit': y_digit, 'parity': y_parity}, epochs=10)

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":

    from tensorflow import keras
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Flatten

    # Carrega MNIST e prepara dois targets: dígito original e paridade (par/ímpar)
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    y_digit  = y_train
    y_parity = y_train % 2

    inputs = Input(shape=(784,))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    output_digit  = Dense(10, activation='softmax', name='digit')(x)
    output_parity = Dense(2, activation='softmax', name='parity')(x)

    model = Model(inputs=inputs, outputs=[output_digit, output_parity])
    model.compile(optimizer='adam',
                  loss={'digit': 'sparse_categorical_crossentropy',
                        'parity': 'sparse_categorical_crossentropy'},
                  loss_weights={'digit': 1.0, 'parity': 0.5},
                  metrics=['accuracy'])

    history = model.fit(X_train, {'digit': y_digit, 'parity': y_parity},
                        epochs=5, validation_split=0.1, verbose=0)

    # Gráfico das curvas de loss para cada saída do modelo multi-output
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['digit_loss'], label='Loss — dígito')
    plt.plot(history.history['parity_loss'], label='Loss — paridade')
    plt.title('Loss por Saída — Modelo Multi-Output')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
