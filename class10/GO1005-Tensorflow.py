# GO1005-Tensorflow
# Verifica a versão instalada do TensorFlow e lista os dispositivos GPU disponíveis,
# útil para confirmar o ambiente de execução antes de treinar modelos.
import tensorflow as tf
print(tf.__version__)  # 2.x.x
print(tf.config.list_physical_devices('GPU'))
