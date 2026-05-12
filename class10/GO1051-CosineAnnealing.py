# GO1051-CosineAnnealing
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

# Calcula o learning rate seguindo um ciclo de cosseno entre initial_lr e min_lr.
# A curva suave evita que o otimizador "pule" o mínimo no final do treino.
def cosine_annealing(epoch, lr, initial_lr=1e-3, min_lr=1e-4, max_epochs=100):
    return min_lr + 0.5 * (initial_lr - min_lr) * \
           (1 + np.cos(np.pi * epoch / max_epochs))

callback = LearningRateScheduler(cosine_annealing)
