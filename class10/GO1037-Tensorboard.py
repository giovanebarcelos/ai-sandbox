# GO1037-Tensorboard
# TensorBoard
# tensorboard_callback = TensorBoard(
#     log_dir='./logs',
#     histogram_freq=1,
#     write_graph=True,
#     write_images=True
# )

# Custom callback para logging
# class CustomLogger(keras.callbacks.Callback):
#     # Exibe no console ao final de cada época os valores de loss e val_loss,
#     # permitindo acompanhar o progresso do treino sem depender do verbose padrão.
#     def on_epoch_end(self, epoch, logs=None):
#         print(f"Epoch {epoch}: loss={logs['loss']:.4f}, "
#               f"val_loss={logs['val_loss']:.4f}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.callbacks import TensorBoard

    # Define CustomLogger que exibe loss e val_loss no console a cada época
    class CustomLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch}: loss={logs['loss']:.4f}, "
                  f"val_loss={logs['val_loss']:.4f}")

    # Treina modelo MNIST com CustomLogger e registra o histórico para plotagem
    (X_train, y_train), _ = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    tensorboard_callback = TensorBoard(log_dir='./GO1037_logs',
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_images=True)

    history = model.fit(X_train, y_train, epochs=5,
                        validation_split=0.1,
                        callbacks=[tensorboard_callback, CustomLogger()],
                        verbose=0)

    # Gráfico das curvas de loss e val_loss monitoradas pelo CustomLogger
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Curvas de Loss — TensorBoard + CustomLogger')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('GO1037-training-log.png', dpi=100, bbox_inches='tight')
    plt.close()
