# GO1029-TensorflowTensorflow
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

# Definir entrada
inputs = Input(shape=(784,))

# Camadas sequenciais
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

# Criar modelo
model = Model(inputs=inputs, outputs=outputs, name='functional_model')
model.compile(...)
model.fit(...)
