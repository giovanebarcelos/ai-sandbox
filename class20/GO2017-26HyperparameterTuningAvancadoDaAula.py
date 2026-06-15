# GO2017-26HyperparameterTuningAvançadoDaAula
!pip install keras-tuner

import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow import keras


def build_model(hp):
    """Model com hiperparâmetros tunáveis"""
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tunar número de camadas
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
        ))

        # Tunar dropout
        if hp.Boolean('dropout'):
            model.add(keras.layers.Dropout(
                rate=hp.Float('dropout_rate', 0, 0.5, step=0.1)
            ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # Tunar learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # Dataset MNIST (subamostrado para o tuning ser rápido em demonstração)
    (X_train_full, y_train_full), (X_val, y_val) = keras.datasets.mnist.load_data()
    X_train_full = X_train_full.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0

    X_train, y_train = X_train_full[:3000], y_train_full[:3000]
    X_val, y_val = X_val[:1000], y_val[:1000]

    # Hyperband tuner (mais eficiente) - parâmetros reduzidos para demo
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=5,
        factor=3,
        directory='tuning',
        project_name='mnist_tuning'
    )

    # Buscar melhores hiperparâmetros
    tuner.search(X_train, y_train,
                 epochs=5,
                 validation_data=(X_val, y_val),
                 callbacks=[keras.callbacks.EarlyStopping(patience=3)])

    # Melhores hiperparâmetros
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"\n🏆 Melhores hiperparâmetros:")
    print(f"  - Layers: {best_hps.get('num_layers')}")
    print(f"  - Units: {[best_hps.get(f'units_{i}') for i in range(best_hps.get('num_layers'))]}")
    print(f"  - Activation: {best_hps.get('activation')}")
    print(f"  - Learning Rate: {best_hps.get('learning_rate'):.6f}")

    # Treinar modelo final
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Gráfico de evolução de loss e accuracy do modelo final
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['loss'], label='Treino')
    axes[0].plot(history.history['val_loss'], label='Validação')
    axes[0].set_title('Loss por época')
    axes[0].set_xlabel('Época')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Treino')
    axes[1].plot(history.history['val_accuracy'], label='Validação')
    axes[1].set_title('Acurácia por época')
    axes[1].set_xlabel('Época')
    axes[1].legend()

    fig.suptitle('Modelo final com melhores hiperparâmetros (Keras Tuner)')
    plt.tight_layout()
    plt.show()
