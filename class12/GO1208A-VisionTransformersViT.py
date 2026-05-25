# GO1208A-34aVisionTransformersVitFuturo
import tensorflow as tf
from tensorflow.keras import layers, Model

class PatchEncoder(layers.Layer):
    """Divide imagem em patches e adiciona embeddings"""
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        # camada Dense projeta cada patch achatado para o espaço de embedding (projection_dim)
        self.projection = layers.Dense(projection_dim)
        # embedding posicional: informa ao transformer a posição de cada patch na imagem
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        # gera índices 0..num_patches-1 para buscar o embedding posicional correspondente
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # soma projeção do conteúdo + posição — o transformer precisa de ambos para entender o contexto
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

    # Passo 1: calcular número de patches — imagem 224×224 com patches 16×16 → 14×14 = 196 patches
    num_patches = (image_size // patch_size) ** 2

    # Passo 2: extrair patches com Conv2D — kernel=patch_size e stride=patch_size garante patches sem sobreposição
    # cada filtro aprende uma projeção linear do patch bruto para o espaço de embedding
    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid"
    )(inputs)

    # Passo 3: achatar dimensões espaciais → sequência 1D de patches (como tokens no NLP)
    patch_dims = patches.shape[-3] * patches.shape[-2]
    patches = layers.Reshape((patch_dims, projection_dim))(patches)

    # Passo 4: codificar patches com projeção linear + embedding posicional
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Passo 5: empilhar transformer_layers blocos encoder (ViT-Base usa 12 blocos)
    for _ in range(transformer_layers):
        # Layer Normalization ANTES da atenção (Pre-LN) — mais estável que Post-LN durante treino
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Multi-Head Attention: cada cabeça aprende a focar em diferentes regiões/relações da imagem
        # key_dim = projection_dim // num_heads divide o espaço entre as cabeças
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim // num_heads
        )(x1, x1)

        # Skip connection 1: soma input + saída da atenção para evitar vanishing gradient
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer Normalization 2 antes do MLP (também Pre-LN)
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP feed-forward: expande para 2× a dimensão e depois comprime — capta padrões complexos
        # GELU: ativação suave que funciona melhor que ReLU em transformers
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dropout(0.1)(x3)  # regularização: evita co-dependência entre neurônios
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)

        # Skip connection 2: soma input do MLP + saída do MLP (segunda conexão residual do bloco)
        encoded_patches = layers.Add()([x3, x2])

    # Normalização final após todos os blocos transformer
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    # Global Average Pooling: agrega informação de todos os 196 patches em um único vetor
    representation = layers.GlobalAveragePooling1D()(representation)

    # MLP head: cabeçalho de classificação — transforma vetor agregado em probabilidades
    features = layers.Dropout(0.3)(representation)  # dropout forte para regularizar
    features = layers.Dense(2048, activation="gelu")(features)
    features = layers.Dropout(0.3)(features)

    # Softmax: converte logits em probabilidades que somam 1 para as num_classes classes
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Instanciar ViT-Base (configuração original do paper): 768-dim, 12 cabeças, 12 blocos
# projection_dim=768: dimensão do espaço de embedding (ViT-Base)
# num_heads=12: divide 768/12 = 64 dims por cabeça de atenção
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

# ─── VISUALIZAÇÃO: DIVISÃO EM PATCHES ───
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

image_size = 224
patch_size = 16
num_patches_side = image_size // patch_size  # 14

# Criar imagem sintética para ilustrar a divisão em patches
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Imagem original com grid de patches
img_demo = np.random.rand(image_size, image_size, 3)
axes[0].imshow(img_demo)
axes[0].set_title(f'Imagem {image_size}×{image_size} dividida em patches', fontsize=13)
axes[0].set_xlabel(f'{num_patches_side} patches por lado → {num_patches_side**2} patches total')
# Desenhar grid de patches
for i in range(0, image_size + 1, patch_size):
    axes[0].axhline(i - 0.5, color='white', linewidth=0.5)
    axes[0].axvline(i - 0.5, color='white', linewidth=0.5)

# Parâmetros por componente do ViT-Base
componentes = ['Patch\nEmbedding', 'Pos.\nEmbedding', 'Multi-Head\nAttention (×12)',
               'MLP\nBlocks (×12)', 'MLP\nHead']
params = [590592, 150528, 28311552, 56689152, 1536000]
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

bars = axes[1].bar(componentes, [p / 1e6 for p in params], color=colors, edgecolor='black')
axes[1].set_title('ViT-Base: Parâmetros por Componente (~86M total)', fontsize=13)
axes[1].set_ylabel('Parâmetros (Milhões)')
for bar, p in zip(bars, params):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{p/1e6:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# ─── VISUALIZAÇÃO: FLUXO DO VISION TRANSFORMER ───
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Fluxo do Vision Transformer (ViT)', fontsize=14, fontweight='bold', pad=15)

etapas = ['Imagem\n224×224', 'Patches\n16×16\n(196 patches)', 'Patch\nEmbedding\n(768-dim)',
          'Transformer\nEncoder\n(×12 blocos)', 'Global Avg\nPooling', 'Classificação\n(1000 classes)']
x_positions = [1, 3, 5, 7, 9, 11]
colors_box = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']

for i, (x, etapa, cor) in enumerate(zip(x_positions, etapas, colors_box)):
    rect = plt.Rectangle((x - 0.7, 2), 1.4, 2, linewidth=1.5,
                          edgecolor='black', facecolor=cor, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, 3, etapa, ha='center', va='center', fontsize=9, fontweight='bold')
    if i < len(x_positions) - 1:
        ax.annotate('', xy=(x_positions[i + 1] - 0.7, 3), xytext=(x + 0.7, 3),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.show()
