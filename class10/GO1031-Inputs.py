# GO1031-Inputs
inputs = Input(shape=(784,))
x = Dense(128, 'relu')(inputs)
x = Dense(64, 'relu')(x)

# Saída 1: Classificação de dígitos
output_digit = Dense(10, 'softmax', name='digit')(x)

# Saída 2: Par ou ímpar
output_parity = Dense(2, 'softmax', name='parity')(x)

model = Model(inputs=inputs, outputs=[output_digit, output_parity])

# Compilar com múltiplas losses
model.compile(
    optimizer='adam',
    loss={
        'digit': 'sparse_categorical_crossentropy',
        'parity': 'sparse_categorical_crossentropy'
    },
    loss_weights={'digit': 1.0, 'parity': 0.5},
    metrics=['accuracy']
)

# Treinar com dois targets
model.fit(X_train, {'digit': y_digit, 'parity': y_parity}, epochs=10)
