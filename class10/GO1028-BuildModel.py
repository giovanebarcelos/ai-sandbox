# GO1028-BuildModel
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))

    # Tunar número de camadas (1-3)
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            # Tunar units (32-512)
            units=hp.Int(f'units_{i}', 32, 512, step=32),
            activation='relu'
        ))
        # Tunar dropout (0.0-0.5)
        model.add(Dropout(hp.Float(f'dropout_{i}', 0, 0.5, 0.1)))

    model.add(Dense(10, 'softmax'))

    # Tunar learning rate
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Busca com Random Search
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='tuner_results',
    project_name='mnist'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=[EarlyStopping(patience=3)]
)

# Melhor modelo
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Best val_accuracy: {tuner.oracle.get_best_trials(1)[0].score}")
print(f"Best LR: {best_hp.get('learning_rate')}")
print(f"Best Units: {best_hp.get('units_0')}")
