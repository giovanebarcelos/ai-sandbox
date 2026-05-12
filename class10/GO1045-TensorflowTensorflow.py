# GO1045-TensorflowTensorflow
# Demonstra transfer learning com VGG16 pré-treinado no ImageNet: congela a base
# convolucional e adiciona novas camadas densas para classificação binária customizada.
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# 1. Carregar modelo pré-treinado (sem top/FC layers)
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Congelar base
base_model.trainable = False

# 3. Adicionar novas camadas
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# 4. Compilar e treinar
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# model.fit(X_train, y_train, epochs=10)

if __name__ == "__main__":
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

    # Usa dados sintéticos 224×224×3 (100 amostras) para treinar com base VGG16 congelada
    np.random.seed(42)
    X_synthetic = np.random.rand(100, 224, 224, 3).astype('float32')
    y_synthetic = np.random.randint(0, 2, 100)

    history = model.fit(X_synthetic, y_synthetic, epochs=3,
                        validation_split=0.2, verbose=1)

    # Gráfico das curvas de loss do transfer learning com VGG16
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss — Transfer Learning VGG16 (base congelada)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
