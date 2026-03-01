# GO1229-TensorflowTensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense


if __name__ == "__main__":
    model = Sequential([
        # 1ª Camada Convolucional
        Conv2D(32, (3,3), activation='relu',
               input_shape=(28, 28, 1)),
        # Output: (26, 26, 32)

        # 2ª Camada Convolucional
        Conv2D(64, (3,3), activation='relu'),
        # Output: (24, 24, 64)

        # Camada de Pooling
        MaxPooling2D(pool_size=(2, 2)),
        # Output: (12, 12, 64)

        # Achatamento para camadas densas
        Flatten(),
        # Output: 9216 neurônios

        # Camadas totalmente conectadas
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes
    ])
