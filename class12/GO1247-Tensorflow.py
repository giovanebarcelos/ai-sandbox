# GO1247-Tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == "__main__":
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    model.fit(datagen.flow(x_train, y_train, batch_size=32), ...)
