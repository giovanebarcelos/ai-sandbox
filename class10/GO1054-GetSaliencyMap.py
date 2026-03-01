# GO1054-GetSaliencyMap
import tensorflow as tf

def get_saliency_map(model, image, class_idx):
    """Calcula saliency map usando gradientes"""
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, 0)

    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        target_class = predictions[:, class_idx]

    gradients = tape.gradient(target_class, image)
    gradients = tf.reduce_max(tf.abs(gradients), axis=-1)
    gradients = gradients.numpy().squeeze()

    # Normalizar 0-1
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min())
    return gradients


if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Sem janela (compatível com Colab)
    import matplotlib.pyplot as plt

    print("=== Demonstração de Saliency Map ===")

    # Modelo simples (sem treinamento – apenas para demo estrutural)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.build((None, 28, 28, 1))

    # Imagem aleatória 28x28 grayscale
    np.random.seed(7)
    image = np.random.rand(28, 28, 1).astype(np.float32)

    saliency = get_saliency_map(model, image, class_idx=0)
    print(f"Shape do saliency map: {saliency.shape}")
    print(f"Valor mín: {saliency.min():.4f}, máx: {saliency.max():.4f}")

    # Salvar visualização
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('Imagem Original')
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title('Saliency Map')
    plt.tight_layout()
    plt.savefig('saliency_demo.png', dpi=80)
    print("Saliency map salvo em saliency_demo.png")
