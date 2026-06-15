# GO2019-28LearningRateSchedulingDaAula
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


# 1. Schedule Manual (decay exponencial)
def lr_schedule(epoch, lr):
    """Reduz LR a cada 10 épocas"""
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.5
    return lr


# 3. Warmup + Cosine Decay (SOTA)
def warmup_cosine_decay(epoch, total_epochs=30, warmup_epochs=5,
                        initial_lr=1e-6, max_lr=1e-3):
    """
    Warmup linear seguido de cosine decay
    Usado em BERT, GPT, Vision Transformers
    """
    if epoch < warmup_epochs:
        # Warmup linear
        return initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max_lr * 0.5 * (1 + np.cos(np.pi * progress))


if __name__ == "__main__":
    TOTAL_EPOCHS = 30

    # Visualizar os schedules de LR antes de treinar
    epocas = np.arange(TOTAL_EPOCHS)

    lr_manual = [1e-3]
    for e in range(1, TOTAL_EPOCHS):
        lr_manual.append(lr_schedule(e, lr_manual[-1]))

    lr_warmup_cosine = [warmup_cosine_decay(e, total_epochs=TOTAL_EPOCHS) for e in epocas]

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, lr_manual, label="Decay manual (x0.5 a cada 10 épocas)")
    plt.plot(epocas, lr_warmup_cosine, label="Warmup + Cosine Decay")
    plt.xlabel("Época")
    plt.ylabel("Learning Rate")
    plt.title("Comparação de estratégias de Learning Rate Scheduling")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Treinar um modelo simples no MNIST aplicando o Warmup + Cosine Decay
    (X_train, y_train), (X_val, y_val) = keras.datasets.mnist.load_data()
    X_train, X_val = X_train.astype("float32") / 255.0, X_val.astype("float32") / 255.0
    X_train, y_train = X_train[:3000], y_train[:3000]
    X_val, y_val = X_val[:1000], y_val[:1000]

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    # 2. ReduceLROnPlateau (automático baseado em métrica) - aplicado junto
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduz LR pela metade
        patience=3,  # Espera 3 épocas sem melhoria
        min_lr=1e-7,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=TOTAL_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[LearningRateScheduler(warmup_cosine_decay), reduce_lr]
    )

    # Gráfico: loss/accuracy ao longo do treino com o schedule aplicado
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss por época (Warmup + Cosine Decay)')
    axes[0].set_xlabel('Época')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Acurácia por época (Warmup + Cosine Decay)')
    axes[1].set_xlabel('Época')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
