# GO1009-ClassificaçãoBinária
# Mostra as configurações de compile() para os três cenários principais: classificação
# binária, multiclasse (MNIST) e regressão, incluindo learning rate customizado.
# Classificação binária
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Classificação multiclasse (MNIST)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Regressão
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Com learning rate customizado
from tensorflow.keras.optimizers import Adam
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Cria modelo simples para classificação binária com dados sintéticos
    model = Sequential([
        Dense(16, activation='relu', input_shape=(10,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Dados sintéticos para treino binário
    X_train = np.random.rand(500, 10).astype('float32')
    y_train = (np.random.rand(500) > 0.5).astype('float32')

    history = model.fit(X_train, y_train, epochs=20,
                        validation_split=0.2, verbose=0)

    # Gráfico da curva de loss ao longo das épocas
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Curva de Loss — Classificação Binária')
    plt.xlabel('Época')
    plt.ylabel('Binary Crossentropy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('GO1009-binary-loss.png', dpi=100, bbox_inches='tight')
    plt.close()
