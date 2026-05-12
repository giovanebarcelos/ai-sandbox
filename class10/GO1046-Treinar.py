# GO1046-Treinar
# Implementa a fase de fine-tuning: descongela as últimas camadas da base VGG16
# e retreina com learning rate muito baixo para ajustar finamente os pesos à nova tarefa.
# 1. Primeiro, treinar com feature extraction (acima)
# ... train for 10 epochs ...

# 2. Descongelar últimas camadas do base_model
# base_model.trainable = True

# Congelar apenas primeiras camadas
# for layer in base_model.layers[:-4]:
#     layer.trainable = False

# 3. Recompilar com LR MUITO BAIXO
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-5),
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# 4. Treinar mais épocas
# model.fit(X_train, y_train, epochs=20)

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

    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow import keras

    # Fase 1: feature extraction com base VGG16 congelada (3 épocas)
    np.random.seed(42)
    X_synthetic = np.random.rand(100, 224, 224, 3).astype('float32')
    y_synthetic = np.random.randint(0, 2, 100)

    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model, Flatten(),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    hist1 = model.fit(X_synthetic, y_synthetic, epochs=3,
                      validation_split=0.2, verbose=0)

    # Fase 2: fine-tuning — descongela últimas 4 camadas e treina com lr=1e-5
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    hist2 = model.fit(X_synthetic, y_synthetic, epochs=3,
                      validation_split=0.2, verbose=0)

    # Gráfico de loss das duas fases com linha vertical separando feature extraction e fine-tuning
    loss_phase1 = hist1.history['loss']
    loss_phase2 = hist2.history['loss']
    all_loss = loss_phase1 + loss_phase2
    n1 = len(loss_phase1)

    plt.figure(figsize=(8, 4))
    plt.plot(range(n1), loss_phase1, label='Feature Extraction', color='#3498DB')
    plt.plot(range(n1, n1 + len(loss_phase2)), loss_phase2,
             label='Fine-Tuning (lr=1e-5)', color='#E67E22')
    plt.axvline(x=n1 - 0.5, color='red', linestyle='--', label='Início do Fine-Tuning')
    plt.title('Loss — Fine-Tuning VGG16')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
