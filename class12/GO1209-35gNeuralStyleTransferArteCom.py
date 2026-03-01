# GO1209-35gNeuralStyleTransferArteCom
# ═══════════════════════════════════════════════════════════════════
# NEURAL STYLE TRANSFER - ARTE COM CNNs
# Transferir estilo de uma imagem para outra
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

print("🎨 NEURAL STYLE TRANSFER")
print("=" * 70)

# ─── 1. CRIAR IMAGENS SINTÉTICAS ───
print("\n🖼️ Criando imagens sintéticas...")

img_size = 224

# Content image (foto simples)
content_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
content_img[:, :] = [100, 150, 200]  # Azul claro
# Adicionar círculo
import cv2
cv2.circle(content_img, (112, 112), 60, (255, 255, 0), -1)  # Amarelo

# Style image (padrão artístico)
style_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
for i in range(0, img_size, 10):
    cv2.line(style_img, (i, 0), (i, img_size), (255, 0, 0), 2)  # Linhas azuis
for i in range(0, img_size, 10):
    cv2.line(style_img, (0, i), (img_size, i), (0, 255, 0), 2)  # Linhas verdes

print(f"  Content: {content_img.shape}")
print(f"  Style: {style_img.shape}")

# ─── 2. CARREGAR VGG19 ───
print("\n🏗️ Carregando VGG19 pré-treinado...")

vgg = VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

print(f"  Layers: {len(vgg.layers)}")

# ─── 3. DEFINIR LAYERS PARA CONTEÚDO E ESTILO ───
# Content: camadas profundas (high-level)
content_layers = ['block5_conv2']

# Style: múltiplas camadas (low + high level)
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                'block4_conv1', 'block5_conv1']

print(f"\n📌 Content layers: {content_layers}")
print(f"📌 Style layers: {style_layers}")

# Criar extractor
outputs = [vgg.get_layer(name).output for name in (content_layers + style_layers)]
extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)

# ─── 4. PREPROCESSING ───
def preprocess(img):
    img = img.astype('float32')
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess(img):
    img = img.copy()
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# ─── 5. EXTRAIR FEATURES ───
print("\n🔍 Extraindo features...")

content_processed = preprocess(content_img)
style_processed = preprocess(style_img)

content_processed = np.expand_dims(content_processed, axis=0)
style_processed = np.expand_dims(style_processed, axis=0)

content_outputs = extractor(content_processed)
style_outputs = extractor(style_processed)

# Separar
content_features = content_outputs[:len(content_layers)]
style_features = style_outputs[len(content_layers):]

print(f"  Content features: {len(content_features)} layers")
print(f"  Style features: {len(style_features)} layers")

# ─── 6. GRAM MATRIX PARA STYLE ───
def gram_matrix(tensor):
    """
    Calcula Gram Matrix para capturar correlações entre features
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Calcular Gram para style
style_grams = [gram_matrix(style_feat) for style_feat in style_features]

# ─── 7. LOSS FUNCTIONS ───
def content_loss(content_target, content_output):
    return tf.reduce_mean(tf.square(content_target - content_output))

def style_loss(style_targets, style_outputs):
    loss = 0
    for target, output in zip(style_targets, style_outputs):
        loss += tf.reduce_mean(tf.square(target - gram_matrix(output)))
    return loss / len(style_targets)

def total_variation_loss(img):
    """Reduz ruído (suaviza imagem)"""
    x_deltas = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_deltas = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

# ─── 8. OTIMIZAÇÃO ───
print("\n🎯 Otimizando imagem...")

# Inicializar com content image
generated_img = tf.Variable(content_processed, dtype=tf.float32)

optimizer = tf.optimizers.Adam(learning_rate=0.02)

# Pesos
content_weight = 1e3
style_weight = 1e-2
tv_weight = 30

print(f"  Content weight: {content_weight}")
print(f"  Style weight: {style_weight}")
print(f"  TV weight: {tv_weight}")

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        outputs = extractor(generated_img)
        gen_content = outputs[:len(content_layers)]
        gen_style = outputs[len(content_layers):]

        loss_c = content_loss(content_features[0], gen_content[0])
        loss_s = style_loss(style_grams, gen_style)
        loss_tv = total_variation_loss(generated_img)

        total_loss = content_weight * loss_c + style_weight * loss_s + tv_weight * loss_tv

    gradients = tape.gradient(total_loss, generated_img)
    optimizer.apply_gradients([(gradients, generated_img)])

    # Clip para range válido
    generated_img.assign(tf.clip_by_value(generated_img, -150, 150))

    return total_loss, loss_c, loss_s, loss_tv

# Treinar
num_iterations = 50
losses = []

for i in range(num_iterations):
    loss, c_loss, s_loss, tv_loss = train_step()
    losses.append(loss.numpy())

    if (i + 1) % 10 == 0:
        print(f"  Iteration {i+1}/{num_iterations}: Loss = {loss:.2f} "
              f"(C: {c_loss:.2f}, S: {s_loss:.2f}, TV: {tv_loss:.2f})")

# ─── 9. RESULTADO ───
print("\n📊 Visualizando resultado...")

result_img = deprocess(generated_img.numpy()[0])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Content Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB))
axes[1].set_title('Style Image', fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
axes[2].set_title('Generated (Stylized)', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.suptitle('Neural Style Transfer', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('style_transfer_result.png', dpi=150)
print("✅ Resultado salvo: style_transfer_result.png")

# Loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.title('Style Transfer Optimization', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('style_transfer_loss.png', dpi=150)
print("✅ Loss salvo: style_transfer_loss.png")

print("\n💡 COMO FUNCIONA:")
print("  1. Content Loss: Features de camadas profundas (conteúdo)")
print("  2. Style Loss: Gram matrix de múltiplas camadas (estilo)")
print("  3. TV Loss: Regularização para suavizar")
print("  4. Otimização: Ajustar pixels para minimizar loss combinada")

print("\n📚 APLICAÇÕES:")
print("  • Arte digital (transferir estilo de pinturas famosas)")
print("  • Filtros de apps (Prisma, DeepArt)")
print("  • Cinema e games (estilização de cenas)")
print("  • Design gráfico")

print("\n✅ STYLE TRANSFER COMPLETO!")
