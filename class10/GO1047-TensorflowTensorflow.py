# GO1047-TensorflowTensorflow
# Define um modelo Sequential com BatchNormalization aplicada antes da ativação,
# estabilizando a distribuição das ativações e permitindo learning rates maiores.
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

model = Sequential([
    Input(shape=(784,)),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(128),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
