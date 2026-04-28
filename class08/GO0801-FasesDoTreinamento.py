# GO0801-FasesDoTreinamento
# IMPLEMENTAÇÃO SIMPLES DE SOM DO ZERO

import numpy as np
import matplotlib.pyplot as plt

class SimpleSOM:
    """
    Self-Organizing Map (SOM) simples
    """
    def __init__(self, map_height, map_width, input_dim, 
                 learning_rate=0.5, sigma=None, n_iterations=1000):
        """
        Parâmetros:
        -----------
        map_height : int
            Altura da grade (número de linhas)
        map_width : int
            Largura da grade (número de colunas)
        input_dim : int
            Dimensão do vetor de entrada
        learning_rate : float
            Taxa de aprendizado inicial
        sigma : float
            Raio inicial de vizinhança (se None, usa max(height, width)/2)
        n_iterations : int
            Número de iterações de treinamento
        """
        self.map_height = map_height
        self.map_width = map_width
        self.input_dim = input_dim
        self.n_iterations = n_iterations
        self.learning_rate_0 = learning_rate

        # Raio inicial
        if sigma is None:
            self.sigma_0 = max(map_height, map_width) / 2.0
        else:
            self.sigma_0 = sigma

        # Inicializar pesos aleatoriamente
        # Shape: (map_height, map_width, input_dim)
        self.weights = np.random.randn(map_height, map_width, input_dim)

        # Normalizar pesos
        for i in range(map_height):
            for j in range(map_width):
                self.weights[i, j] = self.weights[i, j] / np.linalg.norm(self.weights[i, j])

        # Criar grade de coordenadas dos neurônios
        self.neuron_coords = np.array(
            [[np.array([i, j]) for j in range(map_width)] 
             for i in range(map_height)]
        )

    def _decay_function(self, t):
        """
        Calcula learning rate e sigma para iteração t
        """
        # Decaimento exponencial
        learning_rate = self.learning_rate_0 * np.exp(-t / self.n_iterations)
        sigma = self.sigma_0 * np.exp(-t / self.n_iterations)
        return learning_rate, sigma

    def _find_bmu(self, x):
        """
        Encontra Best Matching Unit (BMU) para entrada x

        Returns:
        --------
        (i, j) : tuple
            Coordenadas do BMU na grade
        """
        # Calcular distâncias euclidianas para todos neurônios
        distances = np.linalg.norm(self.weights - x, axis=2)

        # Encontrar índice do mínimo
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)

        return bmu_idx

    def _neighborhood_function(self, bmu_coords, sigma):
        """
        Calcula função de vizinhança gaussiana

        Returns:
        --------
        neighborhood : np.array shape (map_height, map_width)
            Valores de influência para cada neurônio
        """
        # Distância de cada neurônio ao BMU
        distances_sq = np.sum(
            (self.neuron_coords - bmu_coords) ** 2, 
            axis=2
        )

        # Função gaussiana
        neighborhood = np.exp(-distances_sq / (2 * sigma**2))

        return neighborhood

    def _update_weights(self, x, bmu_coords, learning_rate, sigma):
        """
        Atualiza pesos de todos neurônios
        """
        # Calcular influência de vizinhança
        neighborhood = self._neighborhood_function(bmu_coords, sigma)

        # Expandir dimensões para broadcasting
        neighborhood = neighborhood[:, :, np.newaxis]

        # Atualizar pesos
        # w(t+1) = w(t) + α(t) * h(t) * (x - w(t))
        self.weights += learning_rate * neighborhood * (x - self.weights)

    def fit(self, X, verbose=True):
        """
        Treinar SOM com dados X

        Parameters:
        -----------
        X : np.array shape (n_samples, input_dim)
            Dados de treinamento
        """
        n_samples = X.shape[0]

        for t in range(self.n_iterations):
            # Decaimento dos parâmetros
            learning_rate, sigma = self._decay_function(t)

            # Selecionar amostra aleatória
            idx = np.random.randint(0, n_samples)
            x = X[idx]

            # 1. Encontrar BMU
            bmu_coords = self._find_bmu(x)

            # 2. Atualizar pesos
            self._update_weights(x, bmu_coords, learning_rate, sigma)

            # Progresso
            if verbose and (t % 100 == 0 or t == self.n_iterations - 1):
                print(f"Iteração {t+1}/{self.n_iterations} - "
                      f"α={learning_rate:.4f}, σ={sigma:.4f}")

    def predict(self, X):
        """
        Mapeia amostras para coordenadas na grade

        Returns:
        --------
        coords : np.array shape (n_samples, 2)
            Coordenadas (i, j) do BMU para cada amostra
        """
        coords = np.array([self._find_bmu(x) for x in X])
        return coords

    def quantization_error(self, X):
        """
        Calcula erro de quantização (distância média ao BMU)
        """
        errors = []
        for x in X:
            bmu_coords = self._find_bmu(x)
            bmu_weights = self.weights[bmu_coords[0], bmu_coords[1]]
            error = np.linalg.norm(x - bmu_weights)
            errors.append(error)
        return np.mean(errors)

    def plot(self, X, labels=None, title="SOM — Neurônios sobre os Dados"):
        """
        Visualiza os neurônios do SOM sobre os dados originais (apenas 2D).

        Parameters:
        -----------
        X : np.array shape (n_samples, 2)
            Dados de entrada (devem ser 2-dimensionais)
        labels : array-like, opcional
            Rótulos de classe para colorir os pontos
        title : str
            Título do gráfico
        """
        if self.input_dim != 2:
            print("plot() só suporta dados 2D.")
            return

        fig, ax = plt.subplots(figsize=(7, 6))

        # ── dados originais ──────────────────────────────────────────
        if labels is not None:
            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels,
                                 cmap='tab10', alpha=0.4, s=20,
                                 label="Amostras")
        else:
            ax.scatter(X[:, 0], X[:, 1], color='steelblue',
                       alpha=0.4, s=20, label="Amostras")

        # ── grade do SOM (linhas entre neurônios vizinhos) ───────────
        W = self.weights  # shape (H, W, 2)
        # Conexões horizontais
        for i in range(self.map_height):
            for j in range(self.map_width - 1):
                ax.plot([W[i, j, 0], W[i, j+1, 0]],
                        [W[i, j, 1], W[i, j+1, 1]],
                        'k-', linewidth=0.8, alpha=0.5)
        # Conexões verticais
        for i in range(self.map_height - 1):
            for j in range(self.map_width):
                ax.plot([W[i, j, 0], W[i+1, j, 0]],
                        [W[i, j, 1], W[i+1, j, 1]],
                        'k-', linewidth=0.8, alpha=0.5)

        # ── neurônios ────────────────────────────────────────────────
        neurons = W.reshape(-1, 2)
        ax.scatter(neurons[:, 0], neurons[:, 1],
                   color='black', s=60, zorder=5, label="Neurônios")

        ax.set_title(title)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    # Gerar dados sintéticos: 3 clusters em 2D
    n = 100
    X = np.vstack([
        np.random.randn(n, 2) + [2, 2],
        np.random.randn(n, 2) + [-2, 2],
        np.random.randn(n, 2) + [0, -2],
    ])

    print("=== Treinando SOM 5x5 em dados 2D ===")
    som = SimpleSOM(map_height=5, map_width=5, input_dim=2,
                    learning_rate=0.5, n_iterations=500)
    som.fit(X, verbose=True)

    print(f"\nErro de quantização final: {som.quantization_error(X):.4f}")

    # Mapear amostras para a grade
    coords = som.predict(X[:10])
    print("\nCoordenadas BMU das 10 primeiras amostras:")
    for i, (row, col) in enumerate(coords):
        print(f"  Amostra {i}: neurônio ({row}, {col})")

    # Rótulos dos 3 clusters (0, 1, 2)
    labels = np.array([0]*n + [1]*n + [2]*n)

    # Visualizar neurônios do SOM sobre os dados
    som.plot(X, labels=labels, title="SOM 5×5 — Neurônios sobre os 3 Clusters")
