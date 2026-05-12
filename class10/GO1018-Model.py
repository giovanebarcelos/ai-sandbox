# GO1018-Model
# Ilustra a necessidade de incluir a camada Flatten antes das camadas Dense
# ao trabalhar com imagens 2D do MNIST no modelo Sequential.
model = Sequential([
    Flatten(input_shape=(28, 28)),  # ← Adicionar esta linha!
    Dense(128, 'relu'),
    ...
])
