# GO1007-TensorflowTensorflow
# Demonstra a API Funcional do Keras: define o fluxo de dados passando tensores entre
# camadas e cria o modelo a partir de inputs e outputs explícitos com Model().
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
