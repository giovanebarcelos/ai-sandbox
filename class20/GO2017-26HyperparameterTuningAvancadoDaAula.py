# GO2017-26HyperparameterTuningAvançadoDaAula
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    """Model com hiperparâmetros tunáveis"""
    model = keras.Sequential()

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

# Hyperband tuner (mais eficiente)


if __name__ == "__main__":
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=30,
        factor=3,
        directory='tuning',
        project_name='mnist_tuning'
    )

    # Buscar melhores hiperparâmetros
    tuner.search(X_train, y_train, 
                 epochs=30, 
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
    history = best_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
