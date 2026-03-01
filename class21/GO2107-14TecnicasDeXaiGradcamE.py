# GO2107-14TécnicasDeXaiGradcamE
# ═══════════════════════════════════════════════════════════════════
# GRAD-CAM: GRADIENT-WEIGHTED CLASS ACTIVATION MAPPING
# Visualiza quais regiões da imagem influenciaram a decisão da CNN
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Gera heatmap Grad-CAM
    """
    # Criar modelo que retorna activations da última conv + predições
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Calcular gradientes
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient da classe em relação à última conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pooled gradients (importância de cada channel)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizar 0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ─── Exemplo de uso ───
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Carregar modelo


if __name__ == "__main__":
    model = VGG16(weights='imagenet')

    # Carregar e preprocessar imagem
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predição
    preds = model.predict(img_array)
    print("Predições:", decode_predictions(preds, top=3)[0])

    # Gerar Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv3')

    # Superpor heatmap na imagem original
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Blend
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Visualizar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # ═══════════════════════════════════════════════════════════════════
    # LIME: LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS
    # ═══════════════════════════════════════════════════════════════════

    # pip install lime

    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    # Função de predição para LIME
    def predict_fn(images):
        images = np.array([preprocess_input(img) for img in images])
        return model.predict(images)

    # Criar explainer
    explainer = lime_image.LimeImageExplainer()

    # Explicar predição
    explanation = explainer.explain_instance(
        img_array[0].astype('double'),
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )

    # Visualizar
    from skimage.segmentation import mark_boundaries

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title('LIME: Regiões Positivas')
    plt.axis('off')

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title('LIME: Todas as Regiões')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
