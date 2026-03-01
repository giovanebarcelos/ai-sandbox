# GO1248-Tensorflow
from tensorflow.keras.applications import ResNet50


if __name__ == "__main__":
    base = ResNet50(weights='imagenet', include_top=False)
    base.trainable = False  # Congelar
