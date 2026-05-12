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

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    # Gera 3 séries com seed fixa (resultados idênticos) e sem seed (resultados diferentes)
    n = 20
    np.random.seed(42); s1 = np.random.rand(n)
    np.random.seed(42); s2 = np.random.rand(n)
    s3 = np.random.rand(n)  # sem seed: diferente

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(s1, label='Série 1 (seed=42)', marker='o', markersize=4)
    axes[0].plot(s2, label='Série 2 (seed=42)', marker='x', markersize=6, linestyle='--')
    axes[0].set_title('Com seed fixa — resultados sobrepostos')
    axes[0].set_xlabel('Índice')
    axes[0].set_ylabel('Valor aleatório')
    axes[0].legend()

    axes[1].plot(s1, label='Série 1 (seed=42)', marker='o', markersize=4)
    axes[1].plot(s3, label='Série 3 (sem seed)', marker='x', markersize=6, linestyle='--')
    axes[1].set_title('Sem seed fixa — resultados divergem')
    axes[1].set_xlabel('Índice')
    axes[1].set_ylabel('Valor aleatório')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
