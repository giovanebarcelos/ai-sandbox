# GO1040-Tensorflow
# Carrega o dataset Fashion-MNIST diretamente do Keras, pronto para ser usado
# como alternativa mais desafiadora ao MNIST clássico de dígitos.
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    # Nomes das classes do Fashion-MNIST para rotular as amostras
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Gráfico 1: grade 5×2 de amostras do Fashion-MNIST com rótulos das classes
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i], cmap='gray')
        ax.set_title(class_names[y_train[i]], fontsize=9)
        ax.axis('off')
    plt.suptitle('Amostras do Fashion-MNIST', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Normaliza e treina modelo para obter curvas de histórico
    X_train_n = X_train / 255.0
    X_test_n  = X_test / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train_n, y_train, epochs=5,
                        validation_split=0.1, verbose=0)

    # Gráfico 2: curvas de loss e accuracy do Fashion-MNIST
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss — Fashion-MNIST')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Accuracy — Fashion-MNIST')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
