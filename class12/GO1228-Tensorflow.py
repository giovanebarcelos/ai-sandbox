# GO1228-Tensorflow
from tensorflow.keras.layers import Conv2D


if __name__ == "__main__":
    Conv2D(
        filters=32,           # Número de filtros (feature maps)
        kernel_size=(3, 3),   # Tamanho do kernel
        strides=(1, 1),       # Passo de deslocamento
        padding='valid',      # 'valid' ou 'same'
        activation='relu',    # Função de ativação
        input_shape=(28,28,1) # Apenas na 1ª camada
    )
