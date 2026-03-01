# GO1230-Tensorflow
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D
)

# Max Pooling 2×2
MaxPooling2D(pool_size=(2, 2), strides=2)

# Average Pooling
AveragePooling2D(pool_size=(2, 2))

# Global Average Pooling
GlobalAveragePooling2D()
