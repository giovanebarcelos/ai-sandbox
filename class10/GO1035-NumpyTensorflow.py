# GO1035-NumpyTensorflow
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
