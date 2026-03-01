# GO1007-TensorflowTensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
