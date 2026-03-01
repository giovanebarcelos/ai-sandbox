# GO1256-Tensorflow
from tensorflow.keras.regularizers import l2


if __name__ == "__main__":
    Conv2D(64, (3,3), kernel_regularizer=l2(0.01))
    Dense(128, kernel_regularizer=l2(0.01))
