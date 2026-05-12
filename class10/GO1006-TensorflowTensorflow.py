# GO1006-TensorflowTensorflow
# Exemplo de criação de uma rede Sequential com camadas Dense empilhadas,
# ilustrando a forma mais simples de definir uma arquitetura no Keras.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
