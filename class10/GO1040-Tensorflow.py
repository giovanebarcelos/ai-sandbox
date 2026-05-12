# GO1040-Tensorflow
# Carrega o dataset Fashion-MNIST diretamente do Keras, pronto para ser usado
# como alternativa mais desafiadora ao MNIST clássico de dígitos.
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
