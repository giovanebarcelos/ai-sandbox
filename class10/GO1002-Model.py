# GO1002-Model
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    # Carregar MNIST
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Pré-processar: achatar 28×28 → 784, normalizar 0–1, one-hot encode
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    print(f"Treino: {X_train.shape}  |  Teste: {X_test.shape}")

    # Construir modelo
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Treinar
    print("\nTreinando...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Avaliar no conjunto de teste
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nResultados no Teste:")
    print(f"  Loss:     {loss:.4f}")
    print(f"  Acurácia: {acc:.2%}")

    # Mostrar algumas predições
    print("\nExemplos de predição (primeiros 5 do teste):")
    preds = np.argmax(model.predict(X_test[:5], verbose=0), axis=1)
    reais = np.argmax(y_test[:5], axis=1)
    print(f"  Predito: {preds.tolist()}")
    print(f"  Real:    {reais.tolist()}")
