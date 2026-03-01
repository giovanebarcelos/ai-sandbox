# GO1018-Model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # ← Adicionar esta linha!
    Dense(128, 'relu'),
    ...
])
