# GO1245-Tensorflow
from tensorflow.keras.layers import BatchNormalization


if __name__ == "__main__":
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())  # Após Conv, antes/após ReLU
