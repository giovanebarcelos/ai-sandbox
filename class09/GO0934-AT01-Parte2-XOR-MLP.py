# GO0934-AT01-Parte2-XOR-MLP
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE 01 – FUNDAMENTOS DE REDES NEURAIS
# Parte 2: MLP, Forward Propagation, Backpropagation e XOR
# ═══════════════════════════════════════════════════════════════════
#
# OBJETIVO: Implementar do zero um MLP com Backpropagation e provar
# que ele resolve XOR — algo impossível para o Perceptron simples.
#
# BASEADO EM: Aula 09 – Slides 11, 13, 14, 15, 16, 17

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ───────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES (reutilizadas da Parte 1)
# ───────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def mse_loss(y_true, y_pred):
    """Mean Squared Error: J = (1/m) Σ (ŷ - y)²"""
    return np.mean((y_pred - y_true) ** 2)

# ───────────────────────────────────────────────────────────────────
# PASSO 1 – ARQUITETURA DO MLP
# ───────────────────────────────────────────────────────────────────
# Para resolver XOR precisamos de:
#   Camada de entrada: 2 neurônios (x₁, x₂)
#   Camada oculta:     N neurônios + ativação não-linear (sigmoid)
#   Camada de saída:   1 neurônio  + sigmoid (saída 0-1)
#
# FORWARD PROPAGATION:
#   z¹ = W¹·x + b¹      ← pré-ativação da camada oculta
#   a¹ = σ(z¹)          ← ativação da camada oculta
#   z² = W²·a¹ + b²     ← pré-ativação da camada de saída
#   ŷ  = σ(z²)          ← saída final
#
# BACKPROPAGATION (chain rule):
#   δ² = ŷ − y                        ← erro na saída
#   dW² = (a¹)ᵀ · δ² / m             ← gradiente de W²
#   db² = mean(δ², axis=0)            ← gradiente de b²
#   δ¹ = (δ² · (W²)ᵀ) ⊙ σ'(z¹)       ← erro propagado para oculta
#   dW¹ = xᵀ · δ¹ / m                ← gradiente de W¹
#   db¹ = mean(δ¹, axis=0)            ← gradiente de b¹

class MLP:
    """
    Multi-Layer Perceptron para classificação binária.
    Arquitetura: input → [camadas ocultas com sigmoid] → sigmoid na saída.
    Treinamento: Backpropagation + Gradiente Descendente Mini-Batch.
    """

    def __init__(self, layer_sizes, learning_rate=0.5):
        """
        layer_sizes: ex. [2, 4, 1] → 2 inputs, 4 hidden, 1 output
        """
        self.lr          = learning_rate
        self.num_layers  = len(layer_sizes) - 1
        self.losses      = []

        # Inicialização Xavier (boa para sigmoid)
        self.W = []
        self.b = []
        for i in range(self.num_layers):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            std   = np.sqrt(2.0 / (n_in + n_out))   # Xavier
            self.W.append(np.random.randn(n_in, n_out) * std)
            self.b.append(np.zeros((1, n_out)))

    # ── FORWARD ─────────────────────────────────────────────────────
    def forward(self, X):
        """
        Propaga X pela rede.
        Salva z e a de cada camada para usar no backward.
        Retorna a ativação final (ŷ).
        """
        self._z = []   # pré-ativações (z = W·a + b)
        self._a = [X]  # ativações (a[0] = input)

        a = X
        for l in range(self.num_layers):
            z = np.dot(a, self.W[l]) + self.b[l]
            a = sigmoid(z)
            self._z.append(z)
            self._a.append(a)

        return self._a[-1]   # ŷ

    # ── BACKWARD ────────────────────────────────────────────────────
    def backward(self, y):
        """
        Calcula gradientes via chain rule e atualiza W e b.
        Usa os valores de z e a salvos no último forward().
        """
        m = y.shape[0]

        # Erro na camada de saída: δ = ŷ − y
        # (simplificação válida para MSE + sigmoid ou BCE + sigmoid)
        delta = self._a[-1] - y

        for l in reversed(range(self.num_layers)):
            dW = np.dot(self._a[l].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)

            # Atualizar parâmetros (gradiente descendente)
            self.W[l] -= self.lr * dW
            self.b[l]  -= self.lr * db

            # Propagar erro para a camada anterior
            if l > 0:
                delta = np.dot(delta, self.W[l].T) * sigmoid_derivative(self._z[l - 1])

    # ── TREINAMENTO ─────────────────────────────────────────────────
    def fit(self, X, y, epochs=5000, batch_size=None, verbose=True):
        """Treina a rede por `epochs` épocas."""
        m = X.shape[0]
        batch_size = batch_size or m   # default: batch completo

        for epoch in range(epochs):
            idx = np.random.permutation(m)
            for start in range(0, m, batch_size):
                end  = min(start + batch_size, m)
                Xb   = X[idx[start:end]]
                yb   = y[idx[start:end]]
                self.forward(Xb)
                self.backward(yb)

            # Registrar loss total a cada época
            y_hat = self.forward(X)
            loss  = mse_loss(y, y_hat)
            self.losses.append(loss)

            if verbose and (epoch + 1) % 1000 == 0:
                print(f"  Época {epoch+1:5d} | Loss: {loss:.6f}")

        return self

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ───────────────────────────────────────────────────────────────────
# PASSO 2 – TREINAR NO XOR
# ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("PARTE 2 – MLP resolvendo XOR")
print("=" * 60)

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

np.random.seed(42)
mlp = MLP(layer_sizes=[2, 4, 1], learning_rate=0.5)
mlp.fit(X_xor, y_xor, epochs=5000, verbose=True)

acc = mlp.score(X_xor, y_xor)
print(f"\n  Acurácia no XOR: {acc*100:.0f}%")

# Predições detalhadas
y_pred = mlp.forward(X_xor)
print("\n  x₁  x₂ | y_real | ŷ (prob) | classe")
print("  " + "-"*42)
for xi, yi, pi in zip(X_xor, y_xor, y_pred):
    classe = "1" if pi[0] >= 0.5 else "0"
    print(f"   {xi[0]:.0f}   {xi[1]:.0f} |   {yi[0]:.0f}    |  {pi[0]:.4f}   |   {classe}")

# ───────────────────────────────────────────────────────────────────
# PASSO 3 – VISUALIZAÇÕES
# ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.suptitle('Parte 2 – MLP resolve XOR', fontsize=13, fontweight='bold')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Curva de loss
ax1 = fig.add_subplot(gs[0])
ax1.plot(mlp.losses, color='steelblue', linewidth=1.5)
ax1.set_yscale('log')
ax1.set_title('Curva de Aprendizado (Loss × Época)')
ax1.set_xlabel('Época')
ax1.set_ylabel('MSE Loss')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.annotate(f"Loss final\n{mlp.losses[-1]:.6f}",
             xy=(len(mlp.losses)-1, mlp.losses[-1]),
             xytext=(len(mlp.losses)*0.5, mlp.losses[0]*0.3),
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)

# Fronteira de decisão
ax2 = fig.add_subplot(gs[1])
res = 300
xx, yy = np.meshgrid(np.linspace(-0.2, 1.2, res), np.linspace(-0.2, 1.2, res))
grid = np.c_[xx.ravel(), yy.ravel()]
Z    = mlp.forward(grid).reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=50, cmap='RdYlGn', alpha=0.8)
ax2.contour( xx, yy, Z, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')

from matplotlib.patches import Patch
for xi, yi in zip(X_xor, y_xor):
    cor = 'seagreen' if yi[0] == 1 else 'tomato'
    ax2.scatter(xi[0], xi[1], color=cor, s=200, edgecolors='black', zorder=5)
    ax2.annotate(f"({xi[0]:.0f},{xi[1]:.0f})→{yi[0]:.0f}",
                 xy=(xi[0], xi[1]), xytext=(xi[0]+0.07, xi[1]+0.07), fontsize=8)

ax2.legend(handles=[
    Patch(color='seagreen', label='Classe 1'),
    Patch(color='tomato',   label='Classe 0'),
], fontsize=9)
ax2.set_title('Fronteira de Decisão Aprendida')
ax2.set_xlabel('x₁'); ax2.set_ylabel('x₂')

plt.tight_layout()
plt.savefig('GO0934-parte2-xor-mlp.png', dpi=120, bbox_inches='tight')
plt.show()

# ───────────────────────────────────────────────────────────────────
# CHECKPOINT PARTE 2
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ CHECKPOINT PARTE 2")
print("=" * 60)
print(f"  XOR com MLP [2→4→1] → Acc: {acc*100:.0f}%  {'✅ OK' if acc == 1.0 else '⚠️ tente mais épocas'}")
print()
print("💡 CONCLUSÃO: A camada oculta cria representações internas")
print("   que tornam o XOR linearmente separável no espaço oculto.")
print("   O Backpropagation distribui o erro por toda a rede via")
print("   chain rule, ajustando todos os pesos simultaneamente.")
