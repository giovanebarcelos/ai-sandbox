# GO1030-Tensorflow
from tensorflow.keras.layers import Flatten, Concatenate

# Entrada 1: Imagem
input_image = Input(shape=(28, 28), name='image')
x1 = Flatten()(input_image)
x1 = Dense(128, 'relu')(x1)

# Entrada 2: Metadados (ex: contexto adicional)
input_meta = Input(shape=(10,), name='metadata')
x2 = Dense(32, 'relu')(input_meta)

# Concatenar
combined = Concatenate()([x1, x2])

# Camadas finais
x = Dense(64, 'relu')(combined)
outputs = Dense(10, 'softmax')(x)

model = Model(inputs=[input_image, input_meta], outputs=outputs)

# Treinar com dois inputs
model.fit([X_train_images, X_train_meta], y_train, epochs=10)
