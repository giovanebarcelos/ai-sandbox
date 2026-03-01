# GO0901-UsoModerno
# PERCEPTRON - IMPLEMENTAÇÃO DO ZERO
import numpy as np
import matplotlib.pyplot as plt
class Perceptron:
    """Perceptron de camada única para classificação binária"""
    def __init__(self, n_inputs, learning_rate=0.01):
        """
        Inicializa o Perceptron
        Args:
            n_inputs: número de features de entrada
            learning_rate: taxa de aprendizado (η)
        """
        # Inicializar pesos aleatoriamente (pequenos valores)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.errors = []  # Histórico de erros
    def activation(self, z):
        """Função de ativação step (degrau)"""
        return 1 if z >= 0 else 0
    def predict(self, x):
        """Faz predição para uma amostra"""
        # z = w·x + b
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)
    def fit(self, X, y, n_epochs=100):
        """
        Treina o Perceptron
        Args:
            X: matriz de features (n_samples, n_features)
            y: labels (n_samples,)
            n_epochs: número de épocas de treinamento
        """
        for epoch in range(n_epochs):
            errors = 0
            # Para cada amostra de treino
            for xi, target in zip(X, y):
                # 1. Predição
                prediction = self.predict(xi)
                # 2. Calcular erro
                error = target - prediction
                # 3. Atualizar pesos apenas se errou
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += abs(error)
            self.errors.append(errors)
            # Se não errou nenhuma, convergiu!
            if errors == 0:
                print(f"Convergiu na época {epoch+1}!")
                break
        return self
# TESTE: PORTA LÓGICA AND
print("="*60)
print("TESTE: PORTA LÓGICA AND")
print("="*60)
# Dataset AND
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])  # AND: só retorna 1 se ambos forem 1
# Criar e treinar Perceptron
perceptron = Perceptron(n_inputs=2, learning_rate=0.1)
perceptron.fit(X, y, n_epochs=100)
# Testar
print("\nResultados:")
print("x1  x2  | y_true | y_pred")
print("-" * 30)
for xi, target in zip(X, y):
    pred = perceptron.predict(xi)
    print(f"{xi[0]:2.0f}  {xi[1]:2.0f}  |   {target}    |   {pred}")
print(f"\nPesos finais: {perceptron.weights}")
print(f"Bias final: {perceptron.bias:.4f}")
# VISUALIZAR FRONTEIRA DE DECISÃO
plt.figure(figsize=(12, 5))
# Plot 1: Curva de aprendizado
plt.subplot(1, 2, 1)
plt.plot(perceptron.errors, linewidth=2)
plt.xlabel('Época')
plt.ylabel('Número de Erros')
plt.title('Convergência do Perceptron')
plt.grid(True, alpha=0.3)
# Plot 2: Fronteira de decisão
plt.subplot(1, 2, 2)
# Plotar pontos
for i, (xi, yi) in enumerate(zip(X, y)):
    marker = 'o' if yi == 1 else 'x'
    color = 'blue' if yi == 1 else 'red'
    plt.scatter(xi[0], xi[1], marker=marker, s=200, c=color, edgecolors='black', linewidth=2)
# Desenhar linha de decisão: w₁x₁ + w₂x₂ + b = 0
# x₂ = -(w₁x₁ + b) / w₂
w1, w2 = perceptron.weights
b = perceptron.bias
x1_line = np.array([-0.5, 1.5])
x2_line = -(w1 * x1_line + b) / w2
plt.plot(x1_line, x2_line, 'k--', linewidth=2, label='Fronteira de Decisão')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('AND - Fronteira de Decisão')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n✅ Perceptron implementado e testado com sucesso!")
