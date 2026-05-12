# GO1010-Model
# Define uma arquitetura MLP mais profunda para o MNIST com três camadas ocultas
# decrescentes (512→256→128), Dropout para regularização e inicialização He Normal.
# model = Sequential([
#     # Input: 28×28 → 784 neurônios
#     Flatten(input_shape=(28, 28)),
#
#     # Hidden Layer 1: Grande para capturar features
#     Dense(512, activation='relu', kernel_initializer='he_normal'),
#     Dropout(0.3),
#
#     # Hidden Layer 2: Menos neurônios
#     Dense(256, activation='relu', kernel_initializer='he_normal'),
#     Dropout(0.3),
#
#     # Hidden Layer 3: Ainda menos
#     Dense(128, activation='relu', kernel_initializer='he_normal'),
#     Dropout(0.2),
#
#     # Output: 10 classes
#     Dense(10, activation='softmax')
# ], name='MNIST_Classifier')

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout

    # Carrega MNIST e treina a arquitetura profunda por 5 épocas
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ], name='MNIST_Classifier')

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1, verbose=1)

    # Gráfico de curvas de loss e accuracy ao longo das épocas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss por Época (Arquitetura Profunda)')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy por Época (Arquitetura Profunda)')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('GO1010-history.png', dpi=100, bbox_inches='tight')
    plt.close()
