# GO1224-35bGradcamExplainabilityEmCnns
# ═══════════════════════════════════════════════════════════════════
# GRAD-CAM: GRADIENT-WEIGHTED CLASS ACTIVATION MAPPING
# Visualizar quais regiões da imagem influenciam a predição
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import cv2

print("🔍 GRAD-CAM: EXPLAINABILITY EM CNNs")
print("=" * 70)

# ─── 1. CARREGAR MODELO PRÉ-TREINADO ───
print("\n📦 Carregando VGG16 pré-treinado...")

model = VGG16(weights='imagenet')
last_conv_layer_name = 'block5_conv3'

print(f"  Modelo: {model.name}")
print(f"  Input shape: {model.input_shape}")
print(f"  Output: 1000 classes ImageNet")
print(f"  Última conv layer: {last_conv_layer_name}")

# ─── 2. CRIAR IMAGEM DE TESTE ───
print("\n🖼️ Criando imagem sintética de teste...")

img_array = np.zeros((224, 224, 3), dtype=np.float32)

# Adicionar gradiente de cores
for i in range(224):
    for j in range(224):
        img_array[i, j] = [i/224 * 255, j/224 * 255, (i+j)/448 * 255]

# Adicionar formas
cv2.circle(img_array, (112, 112), 60, (255, 0, 0), -1)
cv2.rectangle(img_array, (50, 150), (170, 200), (0, 255, 0), -1)

img_display = img_array.copy() / 255.0

# Preprocessar para VGG16
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

print(f"  Imagem shape: {img_array.shape}")

# ─── 3. FAZER PREDIÇÃO ───
print("\n🎯 Fazendo predição...")

preds = model.predict(img_array, verbose=0)
decoded = decode_predictions(preds, top=3)[0]

print(f"\n  Top-3 Predições:")
for i, (imagenet_id, label, score) in enumerate(decoded):
    print(f"    {i+1}. {label:20s} ({score:.2%})")

target_class = np.argmax(preds[0])

# ─── 4. IMPLEMENTAR GRAD-CAM ───
print("\n🔬 Calculando Grad-CAM...")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera heatmap Grad-CAM
    """
    # Modelo que mapeia input → (last_conv_layer, predictions)
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradientes do output em relação à última conv layer
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling dos gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ponderar feature maps pelos gradientes
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizar [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=target_class)

print(f"  Heatmap shape: {heatmap.shape}")
print(f"  Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

# ─── 5. SOBREPOR HEATMAP NA IMAGEM ───
print("\n🎨 Gerando visualização...")

def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Sobrepõe heatmap na imagem original
    """
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Converter para RGB
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Sobrepor
    img_uint8 = np.uint8(255 * img)
    superimposed = heatmap_colored * alpha + img_uint8 * (1 - alpha)
    superimposed = np.uint8(superimposed)

    return heatmap_colored, superimposed

heatmap_colored, superimposed = overlay_heatmap_on_image(img_display, heatmap)

# ─── 6. VISUALIZAR RESULTADOS ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Linha 1
axes[0, 0].imshow(img_display)
axes[0, 0].set_title('Imagem Original', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(heatmap, cmap='jet')
axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(superimposed)
axes[0, 2].set_title('Sobreposição (α=0.4)', fontsize=14, fontweight='bold')
axes[0, 2].axis('off')

# Linha 2: Diferentes alphas
for idx, alpha in enumerate([0.3, 0.5, 0.7]):
    _, sup = overlay_heatmap_on_image(img_display, heatmap, alpha=alpha)
    axes[1, idx].imshow(sup)
    axes[1, idx].set_title(f'Alpha={alpha}', fontsize=12)
    axes[1, idx].axis('off')

top_pred = decoded[0]
plt.suptitle(f'Grad-CAM Visualization\nPredição: {top_pred[1]} ({top_pred[2]:.1%})', 
            fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('gradcam_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Visualização salva: gradcam_visualization.png")

# ─── 7. GRAD-CAM PARA MÚLTIPLAS CLASSES ───
print("\n📊 Grad-CAM para Top-3 classes...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Imagem original
axes[0].imshow(img_display)
axes[0].set_title('Original', fontsize=14, fontweight='bold')
axes[0].axis('off')

# Top-3 predições
for i in range(3):
    class_idx = np.argsort(preds[0])[::-1][i]
    class_prob = preds[0][class_idx]
    class_name = decode_predictions(preds, top=1000)[0][i][1]

    # Calcular Grad-CAM para esta classe
    heatmap_class = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=class_idx)
    _, superimposed_class = overlay_heatmap_on_image(img_display, heatmap_class)

    axes[i+1].imshow(superimposed_class)
    axes[i+1].set_title(f'#{i+1}: {class_name}\n({class_prob:.1%})', fontsize=11)
    axes[i+1].axis('off')

plt.suptitle('Grad-CAM para Múltiplas Classes (Top-3)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('gradcam_multiclass.png', dpi=150, bbox_inches='tight')
print("✅ Multi-class Grad-CAM salvo: gradcam_multiclass.png")

# ─── 8. ANÁLISE DE INTERPRETABILIDADE ───
print("\n" + "="*70)
print("📊 ANÁLISE DE INTERPRETABILIDADE")
print("="*70)

heatmap_flat = heatmap.flatten()
print(f"\n📈 Estatísticas do Heatmap:")
print(f"  Média: {heatmap_flat.mean():.4f}")
print(f"  Desvio: {heatmap_flat.std():.4f}")
print(f"  Mediana: {np.median(heatmap_flat):.4f}")
print(f"  Pixels ativados (>0.5): {(heatmap_flat > 0.5).sum()} ({(heatmap_flat > 0.5).mean():.1%})")

max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
print(f"\n📍 Região de máxima ativação:")
print(f"  Posição (linha, col): {max_pos}")
print(f"  Valor: {heatmap[max_pos]:.4f}")

print("\n💡 INTERPRETANDO GRAD-CAM:")
print("  🔴 Vermelho: Alta importância para predição")
print("  🟡 Amarelo: Importância moderada")
print("  🔵 Azul: Baixa importância")

print("\n📚 APLICAÇÕES DE GRAD-CAM:")
print("  • Debugging: Por que modelo erra?")
print("  • Confiança: Modelo olha para lugares certos?")
print("  • Bias detection: Atalhos espúrios (ex: watermarks)")
print("  • Medical imaging: Validar atenção clínica")
print("  • Compliance: Explicar decisões (GDPR, regulação)")

print("\n✅ GRAD-CAM COMPLETO!")
