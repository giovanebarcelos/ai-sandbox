# GO1234-Keras
from keras.preprocessing.image import ImageDataGenerator


if __name__ == "__main__":
    datagen = ImageDataGenerator(
        rotation_range=20,        # Rotação ±20°
        width_shift_range=0.2,    # Deslocamento horizontal
        height_shift_range=0.2,   # Deslocamento vertical
        horizontal_flip=True,     # Espelhamento
        zoom_range=0.15,          # Zoom ±15%
        # AlexNet usava PCA color augmentation também
    )

    # 1 imagem → infinitas variações!
