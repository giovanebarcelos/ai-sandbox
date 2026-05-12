# GO1001-MLP
import numpy as np


class MLP:
    # Inicializa a arquitetura da rede: cria matrizes de pesos W e vetores de bias b
    # para cada par de camadas consecutivas. Pesos inicializados com He initialization
    # (escala √(2/n)) para evitar vanishing/exploding gradients com sigmoid.
    def __init__(self, layer_sizes, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.losses = []
        np.random.seed(42)
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [np.zeros((1, s)) for s in layer_sizes[1:]]

    # Função de ativação σ(z) = 1/(1+e^-z): mapeia qualquer valor real para (0,1),
    # introduzindo não-linearidade que permite à rede aprender fronteiras complexas.
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Derivada de σ em função da própria saída: σ'= a(1-a).
    # Usada no backward para calcular o quanto cada neurônio contribuiu para o erro.
    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    # Função de custo MSE: média dos quadrados dos erros entre predito e real.
    # Mede o quão longe a rede está da resposta correta; o objetivo é minimizá-la.
    def _mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    # Passo forward (propagação direta): calcula a saída da rede para uma entrada X.
    # Para cada camada computa z = X·W + b e a = σ(z), guardando todas as ativações
    # intermediárias — necessárias para o backward calcular os gradientes.
    def forward(self, X):
        self.activations = [X]
        current = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = current @ w + b
            current = self._sigmoid(z)
            self.activations.append(current)
        # Camada de saída
        z_out = current @ self.weights[-1] + self.biases[-1]
        output = self._sigmoid(z_out)
        self.activations.append(output)
        return output

    # Passo backward (retropropagação): usa a regra da cadeia para calcular ∂Loss/∂W
    # em cada camada. Começa pelo erro da saída, propaga o delta para trás camada
    # a camada, e atualiza W e b na direção contrária ao gradiente (gradiente descendente).
    def backward(self, y):
        m = y.shape[0]
        deltas = [None] * len(self.weights)

        # Erro na camada de saída
        erro_saida = self.activations[-1] - y
        deltas[-1] = erro_saida * self._sigmoid_deriv(self.activations[-1])

        # Propagar erro para camadas ocultas (trás → frente)
        for i in range(len(self.weights) - 2, -1, -1):
            erro = deltas[i + 1] @ self.weights[i + 1].T
            deltas[i] = erro * self._sigmoid_deriv(self.activations[i + 1])

        # Atualizar pesos e biases com gradiente descendente
        for i in range(len(self.weights)):
            grad_w = self.activations[i].T @ deltas[i] / m
            grad_b = deltas[i].mean(axis=0, keepdims=True)
            self.weights[i] -= self.lr * grad_w
            self.biases[i] -= self.lr * grad_b

    # Loop de treinamento: repete forward → calcula loss → backward por N épocas.
    # A cada iteração a rede ajusta seus pesos para reduzir o erro, convergindo
    # gradualmente para uma solução que generaliza o padrão dos dados.
    def fit(self, X, y):
        self.losses = []
        intervalo = max(1, self.epochs // 5)
        for epoch in range(1, self.epochs + 1):
            y_pred = self.forward(X)
            loss = self._mse(y_pred, y)
            self.losses.append(loss)
            self.backward(y)
            if epoch % intervalo == 0:
                acc = self.score(X, y)
                print(f"  Época {epoch:5d}/{self.epochs} | Loss: {loss:.5f} | Acurácia: {acc:.1%}")
        return self

    # Converte a saída contínua do forward em classe discreta: limiar 0.5 para binário
    # ou argmax para multiclasse. Usado após o treinamento para classificar novos dados.
    def predict(self, X):
        output = self.forward(X)
        if output.shape[1] == 1:
            return (output >= 0.5).astype(int)
        return np.argmax(output, axis=1)

    # Calcula a acurácia: proporção de amostras classificadas corretamente.
    # Métrica de avaliação final — complementa o MSE mostrando o desempenho real.
    def score(self, X, y):
        preds = self.predict(X)
        if y.ndim == 2 and y.shape[1] == 1:
            return np.mean(preds == y.astype(int))
        return np.mean(preds == np.argmax(y, axis=1))


if __name__ == "__main__":
    print("=" * 50)
    print("  MLP NumPy — Problema XOR")
    print("=" * 50)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP([2, 8, 1], learning_rate=0.5, epochs=2000)
    mlp.fit(X, y)

    print("\nResultados:")
    print(f"  {'Entrada':<10} {'Esperado':<10} {'Predito':<10} {'Prob':<8}")
    print("  " + "-" * 40)
    for xi, yi in zip(X, y):
        prob = mlp.forward(xi.reshape(1, -1))[0, 0]
        pred = int(prob >= 0.5)
        status = "✓" if pred == yi[0] else "✗"
        print(f"  {str(xi):<10} {yi[0]:<10} {pred:<10} {prob:.4f}  {status}")

    print(f"\n  Acurácia final: {mlp.score(X, y):.0%}")
    print(f"  Loss final:     {mlp.losses[-1]:.6f}")
