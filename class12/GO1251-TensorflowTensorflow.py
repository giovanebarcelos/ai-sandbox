# GO1251-TensorflowTensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Carregar VGG16 sem top (camadas FC)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelar camadas do VGG16
for layer in base_model.layers:
    layer.trainable = False

# Adicionar novas camadas
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar apenas novas camadas (rápido!)
model.fit(x_train, y_train, epochs=10)

# FINE-TUNING (opcional):
# Descongelar últimas camadas do VGG16
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)
