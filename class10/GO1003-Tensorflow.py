# GO1003-Tensorflow
# Demonstra a execução eager do TensorFlow 2.x: operações em tensores são executadas
# imediatamente, sem sessões ou grafos explícitos.
import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b
print(c)  # Sem sessões! Sem placeholders! Só código normal!
