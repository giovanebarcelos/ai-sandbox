# GO1208A-34aVisionTransformersVitFuturo
import tensorflow as tf
from tensorflow.keras import layers, Model

class PatchEncoder(layers.Layer):
    """Divide imagem em patches e adiciona embeddings"""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_classifier(
    image_size=224,
    patch_size=16,
    num_classes=10,
    projection_dim=768,
    num_heads=12,
    transformer_layers=12
):
    """Cria modelo Vision Transformer"""

    # Input
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Criar patches (14x14 = 196 patches de 16x16)
    num_patches = (image_size // patch_size) ** 2

    # Extrair patches
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid"
    )(inputs)

    patch_dims = patches.shape[-3] * patches.shape[-2]
    patches = layers.Reshape((patch_dims, projection_dim))(patches)

    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer Encoder
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim // num_heads
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)

        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Final layer norm
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    # Global average pooling
    representation = layers.GlobalAveragePooling1D()(representation)

    # MLP head
    features = layers.Dropout(0.3)(representation)
    features = layers.Dense(2048, activation="gelu")(features)
    features = layers.Dropout(0.3)(features)

    # Output
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Criar modelo
model = create_vit_classifier(
    image_size=224,
    patch_size=16,
    num_classes=1000,  # ImageNet
    projection_dim=768,
    num_heads=12,
    transformer_layers=12
)

model.summary()

# Total params: ~86M (ViT-Base)
