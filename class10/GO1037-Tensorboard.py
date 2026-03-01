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
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: loss={logs['loss']:.4f}, "
              f"val_loss={logs['val_loss']:.4f}")
