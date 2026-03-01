# GO1237-Tensorflow
from tensorflow.keras.applications import VGG16

# Carregar VGG sem camadas FC (feature extractor)
base_model = VGG16(weights='imagenet', include_top=False, 
                   input_shape=(224, 224, 3))

# Congelar camadas pre-treinadas
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas específicas do domínio
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
