# GO1205-GradCAMVisualizacaoAtencao
# ═══════════════════════════════════════════════════════════════════
# GRAD-CAM (Gradient-weighted Class Activation Mapping)
# Visualizar ONDE a CNN está "olhando" para fazer predições
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

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
    # Passo 1: criar modelo com duas saídas — ativações da última Conv E predições finais
    # Isso permite capturar tanto os feature maps quanto os gradientes em uma única passagem
    grad_model = keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Passo 2: GradientTape registra operações para calcular derivadas automaticamente
    # Tudo dentro do bloco 'with' é monitorado para backpropagation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Se não especificado, usar classe com maior probabilidade
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        # Score (logit) da classe alvo — é este valor que vamos derivar
        class_channel = predictions[:, pred_index]

    # Passo 3: d(class_score)/d(conv_outputs) — quais ativações mais influenciaram a predição
    grads = tape.gradient(class_channel, conv_outputs)

    # Passo 4: média global dos gradientes por canal (importância de cada filtro)
    # axis=(0,1,2) agrega sobre batch, altura e largura — resultado: um escalar por filtro
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Passo 5: ponderar cada canal do feature map pelo gradiente correspondente
    # pooled_grads[i] indica "o quanto o filtro i importa para a predição"
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Passo 6: colapsar todos os canais em um único mapa de calor (média espacial)
    heatmap = np.mean(conv_outputs, axis=-1)

    # Passo 7: ReLU (zera negativos) + normalização [0,1] — só regiões que ATIVAM a classe importam
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap

# ─── 2. FUNÇÃO PARA SOBREPOR HEATMAP NA IMAGEM ───
def overlay_gradcam(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Sobrepõe heatmap Grad-CAM na imagem original
    """
    # Redimensionar heatmap de (H_conv, W_conv) para (H_img, W_img) — camadas conv têm resolução menor
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Aplicar colormap JET: azul (frio/baixa atenção) → vermelho (quente/alta atenção)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

    # Blending: combina imagem original (peso 1-alpha) com heatmap colorido (peso alpha)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)

    return overlayed, heatmap_colored

# ─── 3. APLICAR GRAD-CAM NO MNIST ───
# Usar modelo já treinado
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test_norm = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Selecionar algumas imagens para demonstrar o Grad-CAM em diferentes dígitos
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

    # Preparar imagem para overlay: Grad-CAM usa BGR (OpenCV) — convém grayscale (1 canal) → RGB (3 canais)
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
# Encontrar todos os índices onde a predição diverge do rótulo real
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
