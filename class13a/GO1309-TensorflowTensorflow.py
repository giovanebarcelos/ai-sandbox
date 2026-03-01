# GO1309-TensorflowTensorflow
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential

# 1. CARREGAR MODELO PRÉ-TREINADO
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. CONGELAR BACKBONE
base_model.trainable = False

# 3. CONSTRUIR MODELO CUSTOM
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# 4. COMPILAR
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. TREINAR (Feature Extraction)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# 6. FINE-TUNING (opcional)
base_model.trainable = True

# Congelar apenas primeiras 100 camadas
for layer in base_model.layers[:100]:
    layer.trainable = False

# Re-compilar com LR menor
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar mais
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20
)
