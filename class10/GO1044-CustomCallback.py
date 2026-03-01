# GO1044-CustomCallback
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.model.save(f'model_epoch_{epoch}.keras')
        if logs.get('loss') != logs.get('loss'):  # NaN check
            self.model.stop_training = True


if __name__ == '__main__':
    import numpy as np
    from tensorflow import keras

    print("=== Demonstração do CustomCallback ===")

    # Modelo simples
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Dados sintéticos
    X = np.random.rand(100, 4).astype(np.float32)
    y = np.random.randint(0, 3, 100)

    print("Treinando com CustomCallback (salva a cada 5 épocas)...")
    model.fit(X, y, epochs=11, callbacks=[CustomCallback()], verbose=1)
