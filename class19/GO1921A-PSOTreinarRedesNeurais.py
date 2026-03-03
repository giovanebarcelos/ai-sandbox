# GO1921A-PSOTreinarRedesNeurais
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NeuralNetPSO:
    """Rede neural 4-5-3 treinada com PSO"""
    def __init__(self, weights):
        # Decodificar pesos (vetor 1D → matrizes)
        # 4 inputs, 5 hidden, 3 outputs
        # Total pesos: (4*5 + 5) + (5*3 + 3) = 25 + 18 = 43

        self.W1 = weights[:20].reshape(4, 5)  # 4x5
        self.b1 = weights[20:25]              # 5
        self.W2 = weights[25:40].reshape(5, 3) # 5x3
        self.b2 = weights[40:43]              # 3

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# Carregar dados
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função objetivo para PSO
def nn_objective(weights):
    """Treinar NN e retornar erro"""
    nn = NeuralNetPSO(weights)
    y_pred = nn.predict(X_train)
    error = 1 - accuracy_score(y_train, y_pred)
    return error

# Otimizar pesos com PSO (43 dimensões)
bounds = [(-2, 2)] * 43  # Pesos iniciais [-2, 2]

pso_nn = PSO(nn_objective, bounds, n_particles=50, max_iter=200, w=0.6, c1=1.8, c2=1.8)
best_weights, best_error = pso_nn.optimize()

# Testar modelo otimizado
final_nn = NeuralNetPSO(best_weights)
y_pred_test = final_nn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"🧠 Rede Neural treinada com PSO:")
print(f"  Train accuracy: {1 - best_error:.4f}")
print(f"  Test accuracy: {test_accuracy:.4f}")

# Comparar com backpropagation (sklearn MLP)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
mlp_accuracy = mlp.score(X_test, y_test)

print(f"📊 Comparação:")
print(f"  PSO: {test_accuracy:.4f}")
print(f"  Backprop (MLP): {mlp_accuracy:.4f}")
