# GO1043-SklearnSklearn
# Carrega o dataset California Housing, divide em treino/teste e normaliza com
# StandardScaler — passo crucial para redes densas aplicadas a problemas de regressão.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar dados
data = fetch_california_housing()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizar (CRUCIAL para regressão!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    import numpy as np

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    # Constrói e treina modelo de regressão (Dense sem softmax, loss=mse) no California Housing
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=30,
                        batch_size=64,
                        validation_split=0.2, verbose=0)

    y_pred = model.predict(X_test, verbose=0).flatten()

    # Gráfico 1: curvas de loss (MSE) treino vs validação
    plt.figure(figsize=(7, 4))
    plt.plot(history.history['loss'], label='Treino MSE')
    plt.plot(history.history['val_loss'], label='Validação MSE')
    plt.title('Loss (MSE) — Regressão California Housing')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Gráfico 2: scatter plot de valores preditos vs valores reais no conjunto de teste
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='#2980B9')
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfeito')
    plt.title('Predito vs Real — California Housing')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.legend()
    plt.tight_layout()
    plt.show()
