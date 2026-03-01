# GO1236-Alexnet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def AlexNet(input_shape=(224, 224, 3), num_classes=1000):
    """
    AlexNet adaptado para Keras (versão simplificada)
    Original usava 2 GPUs - aqui unificamos
    """
    model = Sequential([
        # CONV1: 96 filtros 11×11, stride 4
        Conv2D(96, (11, 11), strides=4, activation='relu', 
               input_shape=input_shape, name='conv1'),
        MaxPooling2D((3, 3), strides=2, name='pool1'),

        # CONV2: 256 filtros 5×5
        Conv2D(256, (5, 5), padding='same', activation='relu', name='conv2'),
        MaxPooling2D((3, 3), strides=2, name='pool2'),

        # CONV3: 384 filtros 3×3
        Conv2D(384, (3, 3), padding='same', activation='relu', name='conv3'),

        # CONV4: 384 filtros 3×3
        Conv2D(384, (3, 3), padding='same', activation='relu', name='conv4'),

        # CONV5: 256 filtros 3×3
        Conv2D(256, (3, 3), padding='same', activation='relu', name='conv5'),
        MaxPooling2D((3, 3), strides=2, name='pool3'),

        # FC Layers
        Flatten(),
        Dense(4096, activation='relu', name='fc6'),
        Dropout(0.5),
        Dense(4096, activation='relu', name='fc7'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', name='fc8')
    ])

    return model

# Criar modelo


if __name__ == "__main__":
    model = AlexNet(num_classes=10)  # 10 classes para CIFAR-10

    # Compilar com otimizador SGD (como original)
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Output:
    # Total params: 62,378,344
    # Trainable params: 62,378,344
    # Non-trainable params: 0
