# GO1035-NumpyTensorflow
# Fixa as seeds de aleatoriedade do Python, NumPy e TensorFlow e ativa operações
# determinísticas na GPU, garantindo reprodutibilidade dos experimentos.
import numpy as np
import tensorflow as tf
import random

# Fixar seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Deterministic ops (GPU)
tf.config.experimental.enable_op_determinism()
