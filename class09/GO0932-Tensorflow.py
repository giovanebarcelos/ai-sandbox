# GO0932-Tensorflow
import tensorflow as tf


if __name__ == "__main__":
    print(tf.__version__)  # Deve ser >= 2.0
    print(tf.config.list_physical_devices('GPU'))
