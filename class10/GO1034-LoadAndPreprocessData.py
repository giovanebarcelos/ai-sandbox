# GO1034-LoadAndPreprocessData
# Separar em funções
# Carrega o dataset MNIST e normaliza os pixels para o intervalo [0, 1],
# retornando os conjuntos de treino e teste prontos para alimentar a rede.
def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, y_train, X_test, y_test

# Cria um modelo Sequential com camada Flatten, Dense com dropout e saída softmax.
# Parâmetros input_shape e num_classes tornam a função reutilizável para diferentes datasets.
def create_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, 'relu'),
        Dropout(0.3),
        Dense(num_classes, 'softmax')
    ])
    return model

# Retorna uma lista de callbacks padrão: EarlyStopping para interromper o treino
# sem melhora e ModelCheckpoint para salvar automaticamente o melhor modelo.
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint('best.keras', save_best_only=True)
    ]

# Main
X_train, y_train, X_test, y_test = load_and_preprocess_data()
model = create_model(input_shape=(28,28), num_classes=10)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2,
                    callbacks=get_callbacks(), epochs=50)
