# GO1030-Tensorflow
# Exemplo de modelo com múltiplas entradas: processa imagem e metadados em ramos
# separados e os concatena antes das camadas finais de classificação.
from tensorflow.keras.layers import Flatten, Concatenate

# # Entrada 1: Imagem
# input_image = Input(shape=(28, 28), name='image')
# x1 = Flatten()(input_image)
# x1 = Dense(128, 'relu')(x1)
#
# # Entrada 2: Metadados (ex: contexto adicional)
# input_meta = Input(shape=(10,), name='metadata')
# x2 = Dense(32, 'relu')(input_meta)
#
# # Concatenar
# combined = Concatenate()([x1, x2])
#
# # Camadas finais
# x = Dense(64, 'relu')(combined)
# outputs = Dense(10, 'softmax')(x)
#
# model = Model(inputs=[input_image, input_meta], outputs=outputs)
#
# # Treinar com dois inputs
# model.fit([X_train_images, X_train_meta], y_train, epochs=10)

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

    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense

    # Dados sintéticos: imagens 28×28 e metadados 10D para treinar modelo multi-input
    np.random.seed(42)
    N = 200
    X_images = np.random.rand(N, 28, 28).astype('float32')
    X_meta   = np.random.rand(N, 10).astype('float32')
    y        = np.random.randint(0, 10, N)

    input_image = Input(shape=(28, 28), name='image')
    x1 = Flatten()(input_image)
    x1 = Dense(128, activation='relu')(x1)

    input_meta = Input(shape=(10,), name='metadata')
    x2 = Dense(32, activation='relu')(input_meta)

    combined = Concatenate()([x1, x2])
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=[input_image, input_meta], outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit([X_images, X_meta], y, epochs=10,
                        validation_split=0.2, verbose=0)

    # Gráfico das curvas de loss do modelo multi-input ao longo das épocas
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss — Modelo Multi-Input')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
