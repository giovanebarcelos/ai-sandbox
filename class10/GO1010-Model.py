# GO1010-Model
model = Sequential([
    # Input: 28×28 → 784 neurônios
    Flatten(input_shape=(28, 28)),

    # Hidden Layer 1: Grande para capturar features
    Dense(512, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.3),

    # Hidden Layer 2: Menos neurônios
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.3),

    # Hidden Layer 3: Ainda menos
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),

    # Output: 10 classes
    Dense(10, activation='softmax')
], name='MNIST_Classifier')
