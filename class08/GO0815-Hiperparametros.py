# GO0815-Hiperparâmetros
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "minisom", "-q"])

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom


if __name__ == "__main__":
    # Carregar e preparar dados
    wine = load_wine()
    scaler = StandardScaler()
    X = scaler.fit_transform(wine.data)
    n_features = X.shape[1]   # 13 features

    som = MiniSom(10, 10, n_features, sigma=5.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 5000)

    new_sample = scaler.transform(wine.data[[0]])   # exemplo: 1ª amostra
    winner = som.winner(new_sample[0])  # Mapear novo dado
    print(f"Novo dado mapeado para neurônio: {winner}")
