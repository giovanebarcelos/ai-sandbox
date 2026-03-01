# GO1205-GradCAMVisualizacaoAtencao
# ═══════════════════════════════════════════════════════════════════
# GRAD-CAM (Gradient-weighted Class Activation Mapping)
# Visualizar ONDE a CNN está "olhando" para fazer predições
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ─── 1. FUNÇÃO GRAD-CAM ───
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera heatmap Grad-CAM para uma imagem

    Args:
        img_array: Imagem de entrada (1, H, W, C)
        model: Modelo CNN
        last_conv_layer_name: Nome da última camada Conv
        pred_index: Índice da classe para visualizar (None = maior prob)

    Returns:
        heatmap: Mapa de calor (H, W)
    """
    # Criar modelo que retorna ativações da última conv + predições
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Calcular gradientes
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Se não especificado, usar classe com maior probabilidade
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # Score da classe alvo
        class_channel = predictions[:, pred_index]

    # Gradientes da classe em relação aos outputs da conv
    grads = tape.gradient(class_channel, conv_outputs)

    # Pooling dos gradientes (importância de cada filtro)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplicar cada canal pelo seu "peso" (importância)
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Heatmap: média de todos os canais
    heatmap = np.mean(conv_outputs, axis=-1)

    # Normalizar entre 0 e 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# ─── 2. FUNÇÃO PARA SOBREPOR HEATMAP NA IMAGEM ───
def overlay_gradcam(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Sobrepõe heatmap Grad-CAM na imagem original
    """
    # Redimensionar heatmap para tamanho da imagem
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Converter heatmap para RGB usando colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

    # Sobrepor
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)

    return overlayed, heatmap_colored

# ─── 3. APLICAR GRAD-CAM NO MNIST ───
# Usar modelo já treinado
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test_norm = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Selecionar algumas imagens
sample_indices = [0, 10, 50, 100, 200]

fig, axes = plt.subplots(len(sample_indices), 4, figsize=(14, 3*len(sample_indices)))

for idx, img_idx in enumerate(sample_indices):
    # Imagem original
    img = x_test[img_idx]
    img_normalized = x_test_norm[img_idx:img_idx+1]
    true_label = y_test[img_idx]

    # Predição
    preds = model.predict(img_normalized, verbose=0)
    pred_label = np.argmax(preds[0])
    pred_conf = preds[0][pred_label]

    # Grad-CAM
    heatmap = make_gradcam_heatmap(
        img_normalized, 
        model, 
        last_conv_layer_name='conv3'
    )

    # Preparar imagem para overlay (converter grayscale para RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlayed, heatmap_colored = overlay_gradcam(img_rgb, heatmap, alpha=0.5)

    # Plotar
    axes[idx, 0].imshow(img, cmap='gray')
    axes[idx, 0].set_title(f'Original\nTrue: {true_label}', fontsize=10)
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(heatmap, cmap='jet')
    axes[idx, 1].set_title(f'Grad-CAM\nPred: {pred_label} ({pred_conf:.2%})', fontsize=10)
    axes[idx, 1].axis('off')

    axes[idx, 2].imshow(overlayed)
    axes[idx, 2].set_title('Overlay', fontsize=10)
    axes[idx, 2].axis('off')

    # Confidence distribution
    axes[idx, 3].bar(range(10), preds[0])
    axes[idx, 3].set_title('Confidence', fontsize=10)
    axes[idx, 3].set_xlabel('Digit')
    axes[idx, 3].set_ylabel('Prob')
    axes[idx, 3].set_xticks(range(10))

plt.suptitle('Grad-CAM: Visualizando Atenção da CNN', fontsize=16)
plt.tight_layout()
plt.show()

# ─── 4. ANÁLISE DE PREDIÇÕES ERRADAS COM GRAD-CAM ───
# Encontrar predições erradas
preds_all = model.predict(x_test_norm, verbose=0)
pred_labels = np.argmax(preds_all, axis=1)
errors = np.where(pred_labels != y_test)[0]

print(f"\nTotal de erros: {len(errors)}")

# Visualizar erros com Grad-CAM
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, ax_row in enumerate(axes):
    if i < len(errors):
        img_idx = errors[i]
        img = x_test[img_idx]
        img_normalized = x_test_norm[img_idx:img_idx+1]
        true_label = y_test[img_idx]

        preds = model.predict(img_normalized, verbose=0)
        pred_label = np.argmax(preds[0])

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_normalized, model, 'conv3')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        overlayed, _ = overlay_gradcam(img_rgb, heatmap, alpha=0.5)

        # Plotar
        ax_row[0].imshow(img, cmap='gray')
        ax_row[0].set_title(f'True: {true_label}', fontsize=12)
        ax_row[0].axis('off')

        ax_row[1].imshow(heatmap, cmap='jet')
        ax_row[1].set_title(f'Pred: {pred_label}', fontsize=12, color='red')
        ax_row[1].axis('off')

        ax_row[2].imshow(overlayed)
        ax_row[2].set_title('Overlay', fontsize=12)
        ax_row[2].axis('off')

        ax_row[3].bar(range(10), preds[0], color=['red' if j==pred_label else 'blue' for j in range(10)])
        ax_row[3].set_title('Confidence', fontsize=12)
        ax_row[3].set_xticks(range(10))

plt.suptitle('Grad-CAM: Análise de Erros', fontsize=16)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# OBSERVAÇÕES:
# • Grad-CAM mostra ONDE a CNN está focando
# • Regiões vermelhas = alta importância
# • Regiões azuis = baixa importância
# • Útil para:
#   - Debugging: verificar se modelo olha para lugares corretos
#   - Interpretabilidade: explicar predições
#   - Detectar bias: modelo focando em features incorretas
# ═══════════════════════════════════════════════════════════════════
