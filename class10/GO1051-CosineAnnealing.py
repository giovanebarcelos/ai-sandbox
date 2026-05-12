# GO1051-CosineAnnealing
import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler

# Calcula o learning rate seguindo um ciclo de cosseno entre initial_lr e min_lr.
# A curva suave evita que o otimizador "pule" o mínimo no final do treino.
def cosine_annealing(epoch, lr, initial_lr=1e-3, min_lr=1e-4, max_epochs=100):
    return min_lr + 0.5 * (initial_lr - min_lr) * \
           (1 + np.cos(np.pi * epoch / max_epochs))

callback = LearningRateScheduler(cosine_annealing)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plota a curva do cosine annealing ao longo de 100 épocas
    max_epochs = 100
    epochs = np.arange(max_epochs)
    lr_values = [cosine_annealing(e, None) for e in epochs]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, lr_values, color='#E74C3C', linewidth=2)
    plt.title('Cosine Annealing — Learning Rate ao longo de 100 Épocas')
    plt.xlabel('Época')
    plt.ylabel('Learning Rate')
    plt.tight_layout()
    plt.savefig('GO1051-cosine-annealing.png', dpi=100, bbox_inches='tight')
    plt.close()
