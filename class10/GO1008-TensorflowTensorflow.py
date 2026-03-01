# GO1008-TensorflowTensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# 1. Carregar dados
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalizar (0-255 → 0-1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 3. Criar modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. Compilar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Treinar
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# 6. Avaliar
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
