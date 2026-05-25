# GO1247-Tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    # model.fit(datagen.flow(x_train, y_train, batch_size=32), ...)

    # ─── VISUALIZAÇÃO: AUGMENTATION NO CIFAR-10 ───
    from tensorflow import keras
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    sample = x_train[3:4].astype('float32') / 255
    true_label = class_names[y_train[3, 0]]

    datagen.fit(sample)

    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    # Original
    axes[0, 0].imshow(sample[0])
    axes[0, 0].set_title(f'Original\n({true_label})', fontsize=9, fontweight='bold', color='green')
    axes[0, 0].axis('off')

    # 19 variações augmentadas
    gen = datagen.flow(sample, batch_size=1)
    for idx in range(1, 20):
        r, c = divmod(idx, 5)
        aug = next(gen)[0]
        axes[r, c].imshow(np.clip(aug, 0, 1))
        axes[r, c].set_title(f'Aug #{idx}', fontsize=8)
        axes[r, c].axis('off')

    plt.suptitle(f'Data Augmentation — 19 Variações da Mesma Imagem CIFAR-10 ({true_label})\n'
                 f'rotation=15, shift=0.1, flip, zoom=0.2, brightness=[0.8,1.2]',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()
