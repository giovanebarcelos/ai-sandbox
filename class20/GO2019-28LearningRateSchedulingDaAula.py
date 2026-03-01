# GO2019-28LearningRateSchedulingDaAula
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

# 1. Schedule Manual (decay exponencial)
def lr_schedule(epoch, lr):
    """Reduz LR a cada 10 épocas"""
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.5
    return lr


if __name__ == "__main__":
    scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 2. ReduceLROnPlateau (automático baseado em métrica)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduz LR pela metade
        patience=5,  # Espera 5 épocas sem melhoria
        min_lr=1e-7,
        verbose=1
    )

    # 3. Warmup + Cosine Decay (SOTA)
    import numpy as np

    def warmup_cosine_decay(epoch, total_epochs=100, warmup_epochs=10, 
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

    # Aplicar
    model.fit(X_train, y_train, 
              epochs=100,
              callbacks=[LearningRateScheduler(warmup_cosine_decay)])
