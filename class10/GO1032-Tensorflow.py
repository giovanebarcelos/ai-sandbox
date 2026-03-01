# GO1032-Tensorflow
from tensorflow.keras.layers import Add

inputs = Input(shape=(784,))
x = Dense(128, 'relu')(inputs)

# Bloco residual
residual = x
x = Dense(128, 'relu')(x)
x = Dense(128, 'relu')(x)
x = Add()([x, residual])  # Skip connection

outputs = Dense(10, 'softmax')(x)
model = Model(inputs, outputs)
