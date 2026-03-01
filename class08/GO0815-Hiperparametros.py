# GO0815-Hiperparâmetros
from minisom import MiniSom


if __name__ == "__main__":
    som = MiniSom(10, 10, n_features, sigma=5.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(X, 5000)
    winner = som.winner(new_sample)  # Mapear novo dado
