# GO1259-Tensorflow
import tensorflow as tf


if __name__ == "__main__":
    print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
    print("TF versão:", tf.__version__)
