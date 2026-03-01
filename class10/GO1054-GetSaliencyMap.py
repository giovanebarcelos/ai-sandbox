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
