# GO1050-Scheduler
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# MÉTODO 1: Callback com função customizada
# Define a taxa de aprendizado em função da época: mantém o valor original nas
# primeiras 10 épocas e reduz 5% por época a partir daí, evitando oscilações tardias.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95

callback = LearningRateScheduler(scheduler)
# model.fit(X_train, y_train, callbacks=[callback])

# MÉTODO 2: Schedule no Optimizer
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# MÉTODO 3: ReduceLROnPlateau (automático!) ⭐
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
# model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[reduce_lr])

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    import numpy as np

    # Plota os 3 schedules de LR ao longo de 50 épocas em um mesmo gráfico
    epochs = np.arange(50)
    initial_lr = 0.01

    # Método 1: scheduler customizado (mantém até época 10, depois -5%/época)
    lr1 = []
    current = initial_lr
    for e in epochs:
        current = scheduler(e, current)
        lr1.append(current)

    # Método 2: ExponentialDecay simulado por época (decay_steps=1 para visualização)
    lr2 = initial_lr * (0.96 ** epochs)

    # Método 3: ReduceLROnPlateau simulado — reduz à metade a cada 5 épocas sem melhora
    lr3 = []
    current = initial_lr
    for e in epochs:
        if e > 0 and e % 5 == 0:
            current = max(current * 0.5, 1e-7)
        lr3.append(current)

    plt.figure(figsize=(9, 4))
    plt.plot(epochs, lr1, label='Método 1: Callback customizado')
    plt.plot(epochs, lr2, label='Método 2: ExponentialDecay')
    plt.plot(epochs, lr3, label='Método 3: ReduceLROnPlateau (simulado)', linestyle='--')
    plt.title('Comparação de LR Schedules ao longo de 50 Épocas')
    plt.xlabel('Época')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
