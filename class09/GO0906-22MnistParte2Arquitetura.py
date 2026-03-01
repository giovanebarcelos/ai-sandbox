# GO0906-22MnistParte2Arquitetura
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: RECONHECIMENTO DE DÍGITOS (MNIST)
# Etapa 2: DEFINIR ARQUITETURA DA REDE
# ═══════════════════════════════════════════════════════════════════

# REDE NEURAL PARA CLASSIFICAÇÃO MULTICLASSE
# FUNÇÕES DE ATIVAÇÃO
def relu(z):
    """ReLU activation"""
    return np.maximum(0, z)
def relu_derivative(z):
    """ReLU derivative"""
    return (z > 0).astype(float)
def softmax(z):
    """Softmax para classificação multiclasse"""
    # Trick para estabilidade numérica
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
# REDE NEURAL MULTICLASSE
class MulticlassNN:
    """Neural Network para classificação multiclasse"""
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        Args:
            input_size: número de features
            hidden_sizes: lista com neurônios em cada camada oculta
            output_size: número de classes
            learning_rate: taxa de aprendizado
        """
        self.lr = learning_rate
        # Arquitetura
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1
        # Inicializar pesos (He initialization para ReLU)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(w)
            self.biases.append(b)
        # Histórico
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    def forward(self, X):
        """Forward propagation"""
        self.z_values = []
        self.a_values = [X]
        # Camadas ocultas (ReLU)
        for i in range(self.num_layers - 1):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            a = relu(z)
            self.z_values.append(z)
            self.a_values.append(a)
        # Camada de saída (Softmax)
        z = np.dot(self.a_values[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)
        self.z_values.append(z)
        self.a_values.append(a)
        return self.a_values[-1]
    def backward(self, X, y):
        """Backpropagation"""
        m = X.shape[0]
        # Gradiente da camada de saída (softmax + cross-entropy)
        delta = self.a_values[-1] - y
        # Backprop através das camadas
        for i in reversed(range(self.num_layers)):
            # Gradientes dos parâmetros
            dW = np.dot(self.a_values[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            # Atualizar parâmetros
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
            # Propagar para camada anterior
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.z_values[i-1])
    def train_epoch(self, X, y, batch_size=32):
        """Treinar uma época"""
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            self.forward(X_batch)
            self.backward(X_batch, y_batch)
    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=True):
        """Treinar a rede"""
        for epoch in range(epochs):
            # Treinar
            self.train_epoch(X_train, y_train, batch_size)
            # Avaliar
            train_pred = self.forward(X_train)
            train_loss = categorical_cross_entropy(y_train, train_pred)
            train_acc = self.accuracy(X_train, y_train)
            val_pred = self.forward(X_val)
            val_loss = categorical_cross_entropy(y_val, val_pred)
            val_acc = self.accuracy(X_val, y_val)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Época {epoch+1}/{epochs}")
                print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")
    def predict(self, X):
        """Predizer classes"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    def accuracy(self, X, y):
        """Calcular acurácia"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)


if __name__ == "__main__":
    print("\n3. Definindo arquitetura da rede...")
    print("   Input: 784 neurons (28×28 pixels)")
    print("   Hidden 1: 128 neurons (ReLU)")
    print("   Hidden 2: 64 neurons (ReLU)")
    print("   Output: 10 neurons (Softmax)")
    print("\n✅ Arquitetura definida!")

    # ───────────────────────────────────────────────────────────────────
    # ✅ CHECKPOINT ETAPA 2:
    # ───────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("✅ CHECKPOINT ETAPA 2 - VALIDAÇÕES")
    print("="*70)

    # Criar instância temporária para testar
    test_nn = MulticlassNN(input_size=784, hidden_sizes=[128, 64], output_size=10)

    # Testar forward com batch pequeno
    test_output = test_nn.forward(X_train[:10])

    print(f"✓ Classe MulticlassNN: OK")
    print(f"✓ Arquitetura: 784 → 128 → 64 → 10")
    print(f"✓ Forward pass: OK (output shape = {test_output.shape})")
    print(f"✓ Softmax: OK (soma por linha = {test_output[0].sum():.4f} ≈ 1.0)")
    print(f"✓ Total de parâmetros: {sum(w.size for w in test_nn.weights) + sum(b.size for b in test_nn.biases):,}")

    # Validar He initialization
    first_layer_weights = test_nn.weights[0]
    expected_std = np.sqrt(2.0 / 784)
    actual_std = first_layer_weights.std()
    print(f"✓ He init: std esperado={expected_std:.4f}, std real={actual_std:.4f} (OK se próximos)")

    print("\n🎉 Etapa 2 completa! Prossiga para Slide 23 (Etapa 3)")
