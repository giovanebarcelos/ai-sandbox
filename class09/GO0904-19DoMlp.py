# GO0904-19DoMlp
# MLP COMPLETO COM BACKPROPAGATION
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# FUNÇÕES AUXILIARES
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
def binary_cross_entropy(y_true, y_pred):
    """Binary cross-entropy loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
# CLASSE MLP PROFISSIONAL
class NeuralNetwork:
    """
    Multi-Layer Perceptron com suporte para múltiplas camadas ocultas
    """
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Args:
            layer_sizes: lista com número de neurônios por camada
                        Ex: [2, 4, 3, 1] = input:2, hidden:4,3, output:1
            learning_rate: taxa de aprendizado
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lr = learning_rate
        # Inicializar pesos e bias
        self.weights = []
        self.biases = []
        # Xavier initialization
        for i in range(self.num_layers - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(w)
            self.biases.append(b)
        # Histórico
        self.train_losses = []
        self.val_losses = []
    def forward(self, X):
        """Forward propagation"""
        self.z_values = []  # Pré-ativações
        self.a_values = [X]  # Ativações (a[0] = input)
        for i in range(self.num_layers - 1):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            a = sigmoid(z)
            self.z_values.append(z)
            self.a_values.append(a)
        return self.a_values[-1]
    def backward(self, X, y):
        """Backpropagation"""
        m = X.shape[0]
        # Gradientes
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)
        # Camada de saída
        delta = self.a_values[-1] - y
        # Backprop através de cada camada
        for i in reversed(range(self.num_layers - 1)):
            # Gradientes dos parâmetros
            dW[i] = np.dot(self.a_values[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
            # Propagar erro para camada anterior
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.z_values[i-1])
        # Atualizar parâmetros
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.lr * dW[i]
            self.biases[i] -= self.lr * db[i]
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=1000, batch_size=32, verbose=True):
        """
        Treinar a rede
        Args:
            X_train: features de treino
            y_train: labels de treino
            X_val: features de validação (opcional)
            y_val: labels de validação (opcional)
            epochs: número de épocas
            batch_size: tamanho do mini-batch
            verbose: imprimir progresso
        """
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            # Embaralhar dados
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            # Mini-batch gradient descent
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                # Forward + Backward
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            # Calcular losses
            train_pred = self.forward(X_train)
            train_loss = binary_cross_entropy(y_train, train_pred)
            self.train_losses.append(train_loss)
            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = binary_cross_entropy(y_val, val_pred)
                self.val_losses.append(val_loss)
            # Log
            if verbose and (epoch + 1) % 100 == 0:
                msg = f"Época {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
    def predict(self, X):
        """Fazer predições"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
    def score(self, X, y):
        """Calcular acurácia"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
# TESTE COM DATASET "MOONS"


if __name__ == "__main__":
    print("="*70)
    print("TREINANDO MLP NO DATASET MOONS")
    print("="*70)
    # Gerar dataset
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # Criar e treinar rede
    # Arquitetura: 2 inputs → 8 hidden → 4 hidden → 1 output
    nn = NeuralNetwork(layer_sizes=[2, 8, 4, 1], learning_rate=0.1)
    nn.fit(X_train, y_train, X_val, y_val, epochs=1000, batch_size=16, verbose=True)
    # Avaliar
    train_acc = nn.score(X_train, y_train)
    val_acc = nn.score(X_val, y_val)
    print(f"\n{'='*70}")
    print(f"RESULTADOS FINAIS")
    print(f"{'='*70}")
    print(f"Acurácia Treino:    {train_acc*100:.2f}%")
    print(f"Acurácia Validação: {val_acc*100:.2f}%")
    # VISUALIZAR
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plot 1: Curvas de aprendizado
    axes[0].plot(nn.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(nn.val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss (Binary Cross-Entropy)')
    axes[0].set_title('Curvas de Aprendizado')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Plot 2: Fronteira de decisão
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = nn.forward(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    axes[1].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    axes[1].scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', 
                   edgecolors='black', linewidth=1, s=50)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].set_title(f'Fronteira de Decisão (Val Acc: {val_acc*100:.1f}%)')
    plt.tight_layout()
    plt.show()
    print("\n✅ MLP completo implementado e testado!")
