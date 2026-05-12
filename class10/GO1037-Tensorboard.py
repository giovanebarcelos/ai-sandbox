# GO1037-Tensorboard
# TensorBoard
tensorboard_callback = TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Custom callback para logging
class CustomLogger(keras.callbacks.Callback):
    # Exibe no console ao final de cada época os valores de loss e val_loss,
    # permitindo acompanhar o progresso do treino sem depender do verbose padrão.
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: loss={logs['loss']:.4f}, "
              f"val_loss={logs['val_loss']:.4f}")
