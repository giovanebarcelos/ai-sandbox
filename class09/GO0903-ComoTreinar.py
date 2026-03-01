# GO0903-ComoTreinar
# RESOLVER XOR COM MLP DE 2 CAMADAS
import numpy as np
import matplotlib.pyplot as plt
# FUNÇÕES DE ATIVAÇÃO
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip para evitar overflow
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
# CLASSE MLP
class MLP:
    """Multi-Layer Perceptron com 1 camada oculta"""
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate=0.5):
        """
        Args:
            n_inputs: número de features de entrada
            n_hidden: número de neurônios na camada oculta
            n_outputs: número de neurônios na saída
            learning_rate: taxa de aprendizado
        """
        self.lr = learning_rate
        # Inicializar pesos aleatoriamente (Xavier initialization)
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_outputs))
        self.losses = []
    def forward(self, X):
        """Forward propagation"""
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    def backward(self, X, y):
        """Backpropagation"""
        m = X.shape[0]
        # Gradiente da camada de saída
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        # Gradiente da camada oculta
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        # Atualizar pesos
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    def train(self, X, y, n_epochs=10000):
        """Treinar a rede"""
        for epoch in range(n_epochs):
            # Forward
            output = self.forward(X)
            # Calcular loss (MSE)
            loss = np.mean((output - y)**2)
            self.losses.append(loss)
            # Backward
            self.backward(X, y)
            # Log progress
            if (epoch + 1) % 1000 == 0:
                print(f"Época {epoch+1}/{n_epochs}, Loss: {loss:.6f}")
    def predict(self, X):
        """Fazer predições"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
# TREINAR MLP NO XOR
print("="*60)
print("RESOLVENDO XOR COM MLP")
print("="*60)
# Dataset XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR
# Criar e treinar MLP
mlp = MLP(n_inputs=2, n_hidden=2, n_outputs=1, learning_rate=0.5)
mlp.train(X, y, n_epochs=10000)
# Testar
print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print("x1  x2  | y_true | y_pred | output")
print("-" * 45)
predictions = mlp.predict(X)
outputs = mlp.forward(X)
for xi, yi, pred, out in zip(X, y, predictions, outputs):
    print(f"{xi[0]:2.0f}  {xi[1]:2.0f}  |   {yi[0]}    |   {pred[0]}    | {out[0]:.4f}")
print(f"\nAcurácia: {np.mean(predictions == y) * 100:.1f}%")
# VISUALIZAR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Plot 1: Curva de aprendizado
axes[0].plot(mlp.losses, linewidth=2)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Convergência do MLP')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)
# Plot 2: Fronteira de decisão
axes[1].set_title('XOR - Fronteira de Decisão do MLP')
axes[1].set_xlabel('x₁')
axes[1].set_ylabel('x₂')
# Criar grid para visualizar fronteira
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100),
                     np.linspace(-0.5, 1.5, 100))
Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plotar contorno
axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
# Plotar pontos
for xi, yi in zip(X, y):
    marker = 'o' if yi[0] == 1 else 'x'
    color = 'blue' if yi[0] == 1 else 'red'
    axes[1].scatter(xi[0], xi[1], marker=marker, s=200, c=color,
                   edgecolors='black', linewidth=2, zorder=5)
axes[1].set_xlim(-0.5, 1.5)
axes[1].set_ylim(-0.5, 1.5)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n✅ MLP resolveu XOR com sucesso!")
