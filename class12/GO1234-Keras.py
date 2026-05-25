# GO1234-Keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    # Cada parâmetro simula uma transformação física realista da imagem:
    # O modelo aprende invariabilidade a esses tipos de transformação
    datagen = ImageDataGenerator(
        rotation_range=20,        # Rotação ±20° — objeto pode estar inclinado
        width_shift_range=0.2,    # Deslocamento horizontal ±20% — objeto não centrado
        height_shift_range=0.2,   # Deslocamento vertical ±20% — idem
        horizontal_flip=True,     # Espelhamento — simetria (ex: gato de frente/costas)
        zoom_range=0.15,          # Zoom ±15% — objeto mais perto ou longe
        # AlexNet usava PCA color augmentation também
    )

    # 1 imagem → infinitas variações! Cada epoch usa uma variação diferente

    # ─── VISUALIZAÇÃO: EFEITOS DOS PARÂMETROS DE AUGMENTATION ───
    from tensorflow import keras
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    sample = x_train[0:1].astype('float32') / 255

    augmentations = [
        ('Original',          ImageDataGenerator()),
        ('rotation_range=30', ImageDataGenerator(rotation_range=30, fill_mode='nearest')),
        ('width_shift=0.3',   ImageDataGenerator(width_shift_range=0.3, fill_mode='nearest')),
        ('height_shift=0.3',  ImageDataGenerator(height_shift_range=0.3, fill_mode='nearest')),
        ('horizontal_flip',   ImageDataGenerator(horizontal_flip=True)),
        ('zoom_range=0.3',    ImageDataGenerator(zoom_range=0.3, fill_mode='nearest')),
        ('shear_range=0.3',   ImageDataGenerator(shear_range=0.3, fill_mode='nearest')),
        ('brightness [0.5,1.5]', ImageDataGenerator(brightness_range=[0.5, 1.5])),
        ('Tudo junto (AlexNet)', ImageDataGenerator(
            rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
            horizontal_flip=True, zoom_range=0.15)),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (name, gen) in zip(axes.flat, augmentations):
        gen.fit(sample)
        aug_img = next(gen.flow(sample, batch_size=1))[0]
        ax.imshow(np.clip(aug_img, 0, 1))
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.axis('off')

    plt.suptitle('ImageDataGenerator (AlexNet) — Efeito de Cada Augmentation\n(CIFAR-10, imagem original no canto superior esquerdo)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
