# GO1260-Tensorflow
# Treina 2x mais rápido em GPUs modernas
from tensorflow.keras.mixed_precision import set_global_policy


if __name__ == "__main__":
    set_global_policy('mixed_float16')
