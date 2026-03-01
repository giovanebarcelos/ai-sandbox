# GO1049-BuildModel
import keras_tuner as kt
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Tunar número de camadas (1-3)
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            # Tunar número de neurônios (32-512)
            units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
            activation='relu'
        ))
        # Tunar dropout (0.0-0.5)
        model.add(keras.layers.Dropout(
            rate=hp.Float('dropout', 0.0, 0.5, step=0.1)
        ))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # Tunar learning rate
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 2. Criar tuner (Hyperband)
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='mnist_tuning'
)

# 3. Executar busca
tuner.search(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)

# 4. Recuperar resultados
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best LR: {best_hps.get('learning_rate')}")
print(f"Best units: {best_hps.get('units_0')}")

# 5. Treinar modelo final
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))
