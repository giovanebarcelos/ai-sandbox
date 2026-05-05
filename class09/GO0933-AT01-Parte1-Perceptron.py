# GO0933-AT01-Parte1-Perceptron
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE 01 – FUNDAMENTOS DE REDES NEURAIS
# Parte 1: Neurônio Artificial, Funções de Ativação e Perceptron
# ═══════════════════════════════════════════════════════════════════
#
# OBJETIVO: Implementar do zero o neurônio artificial e o Perceptron,
# verificar convergência em AND/OR e observar a falha em XOR.
#
# BASEADO EM: Aula 09 – Slides 5, 6, 8, 10

import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────
# PASSO 1 – FUNÇÕES DE ATIVAÇÃO
# ───────────────────────────────────────────────────────────────────
# O neurônio artificial calcula: z = w·x + b   e   y = f(z)
# A função f é chamada "função de ativação".

def step(z):
    """Step (degrau): retorna 1 se z >= 0, senão 0. Usada no Perceptron."""
    return (z >= 0).astype(float)

def sigmoid(z):
    """Sigmoid: comprime qualquer valor para (0, 1). σ(z) = 1 / (1 + e^-z)"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    """Derivada da sigmoid: σ'(z) = σ(z) · (1 − σ(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z):
    """ReLU: max(0, z). Padrão para camadas ocultas em deep learning."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivada do ReLU: 1 se z > 0, senão 0."""
    return (z > 0).astype(float)

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR AS FUNÇÕES DE ATIVAÇÃO
# ───────────────────────────────────────────────────────────────────
z = np.linspace(-5, 5, 300)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Parte 1 – Funções de Ativação', fontsize=13, fontweight='bold')

axes[0].plot(z, step(z),    color='steelblue', linewidth=2)
axes[0].set_title('Step (Degrau)\nUsada no Perceptron')
axes[0].set_xlabel('z'); axes[0].grid(True, alpha=0.3)

axes[1].plot(z, sigmoid(z), color='tomato', linewidth=2)
axes[1].set_title('Sigmoid\nSaída: (0, 1)')
axes[1].set_xlabel('z'); axes[1].grid(True, alpha=0.3)

axes[2].plot(z, relu(z),    color='seagreen', linewidth=2)
axes[2].set_title('ReLU\nPadrão nas camadas ocultas')
axes[2].set_xlabel('z'); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('GO0933-parte1-funcoes-ativacao.png', dpi=120, bbox_inches='tight')
plt.show()

# ───────────────────────────────────────────────────────────────────
# PASSO 2 – CLASSE PERCEPTRON
# ───────────────────────────────────────────────────────────────────
# O Perceptron é o neurônio mais simples:
#   1. Calcula z = w·x + b
#   2. Aplica step: ŷ = step(z)
#   3. Atualiza pesos: w ← w + η·(y_true − ŷ)·x

class Perceptron:
    """
    Perceptron de Frank Rosenblatt (1958).
    Aprende fronteiras de decisão LINEARES.
    Regra de atualização: w ← w + η · (y_true − ŷ) · x
    """

    def __init__(self, n_inputs, learning_rate=0.1, max_epochs=100):
        self.lr     = learning_rate
        self.epochs = max_epochs
        # Pesos inicializados com zeros
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.errors_per_epoch = []

    def predict(self, X):
        """Prediz classe (0 ou 1) para cada amostra em X."""
        z = np.dot(X, self.w) + self.b
        return step(z)

    def fit(self, X, y):
        """Treina o Perceptron usando a regra de atualização de Rosenblatt."""
        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = step(np.dot(xi, self.w) + self.b)
                error  = yi - y_pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:
                print(f"  Convergiu na época {epoch + 1}!")
                break
        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ───────────────────────────────────────────────────────────────────
# PASSO 3 – TESTAR EM AND, OR E XOR
# ───────────────────────────────────────────────────────────────────
datasets = {
    'AND': (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([0, 0, 0, 1], dtype=float)
    ),
    'OR': (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([0, 1, 1, 1], dtype=float)
    ),
    'XOR': (
        np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float),
        np.array([0, 1, 1, 0], dtype=float)
    ),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Parte 1 – Perceptron: AND / OR / XOR', fontsize=13, fontweight='bold')

results = {}
for col, (name, (X, y)) in enumerate(datasets.items()):
    p = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=100)
    print(f"\n── Treinando Perceptron no {name} ──")
    p.fit(X, y)
    acc = p.score(X, y)
    results[name] = acc
    print(f"  Acurácia: {acc * 100:.0f}%")

    # Curva de erros
    axes[0, col].plot(p.errors_per_epoch, color='steelblue', marker='o', markersize=4)
    axes[0, col].set_title(f'{name} – Erros por época')
    axes[0, col].set_xlabel('Época'); axes[0, col].set_ylabel('Erros')
    axes[0, col].grid(True, alpha=0.3)

    # Fronteira de decisão
    ax = axes[1, col]
    if p.w[1] != 0:
        x_line = np.array([-0.2, 1.2])
        y_line = -(p.w[0] * x_line + p.b) / p.w[1]
        ax.plot(x_line, y_line, 'k--', linewidth=1.5, label='Fronteira')

    colors = ['tomato' if yi == 0 else 'seagreen' for yi in y]
    for xi, ci in zip(X, colors):
        ax.scatter(xi[0], xi[1], color=ci, s=200, edgecolors='black', zorder=5)
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
    ax.set_title(f'{name} – Fronteira  |  Acc: {acc*100:.0f}%')
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('GO0933-parte1-perceptron-fronteiras.png', dpi=120, bbox_inches='tight')
plt.show()

# ───────────────────────────────────────────────────────────────────
# CHECKPOINT PARTE 1
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ CHECKPOINT PARTE 1")
print("=" * 60)
print(f"  AND  → Acc: {results['AND']*100:.0f}%  {'✅ OK' if results['AND'] == 1.0 else '❌'}")
print(f"  OR   → Acc: {results['OR']*100:.0f}%  {'✅ OK' if results['OR'] == 1.0 else '❌'}")
print(f"  XOR  → Acc: {results['XOR']*100:.0f}%  {'✅ OK (deve falhar!)' if results['XOR'] < 1.0 else '⚠️ inesperado'}")
print()
print("💡 CONCLUSÃO: O Perceptron converge em AND e OR (linearmente")
print("   separáveis), mas FALHA em XOR (não linearmente separável).")
print("   Solução: adicionar camada oculta → MLP com Backpropagation!")
