# GO1003-Tensorflow
import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = a + b
print(c)  # Sem sessões! Sem placeholders! Só código normal!
