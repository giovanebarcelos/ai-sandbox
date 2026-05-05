# GO0935-AT01-Parte3-MNIST
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE 01 – FUNDAMENTOS DE REDES NEURAIS
# Parte 3: Classificador de Dígitos MNIST do Zero
# ═══════════════════════════════════════════════════════════════════
#
# OBJETIVO: Aplicar todos os conceitos anteriores em um problema real.
# Treinar uma MLP para reconhecer dígitos manuscritos (0–9) do MNIST.
#
# BASEADO EM: Aula 09 – Slides 21A, 22, 23, 24

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ───────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ───────────────────────────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    """Softmax estável numericamente: saída = distribuição de probabilidade."""
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    """Categorical cross-entropy: J = −(1/m) Σ Σ y·log(ŷ)"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

# ───────────────────────────────────────────────────────────────────
# PASSO 1 – CARREGAR E PREPARAR OS DADOS
# ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("PARTE 3 – Classificador MNIST")
print("=" * 60)
print("\n1. Carregando MNIST...")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y  = mnist.data, mnist.target.astype(int)

# Normalização: pixel 0–255 → 0.0–1.0
X = X / 255.0
print(f"   Shape: {X.shape}  |  Classes: {np.unique(y)}")

# Split estratificado: 6 000 treino | 2 000 val | 2 000 teste
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, train_size=6000, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

y_train_oh = one_hot(y_train)
y_val_oh   = one_hot(y_val)
y_test_oh  = one_hot(y_test)

print(f"   Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

# Visualizar 10 amostras
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Parte 3 – Amostras MNIST', fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}'); ax.axis('off')
plt.tight_layout()
plt.savefig('GO0935-parte3-amostras.png', dpi=120, bbox_inches='tight')
plt.show()

# ───────────────────────────────────────────────────────────────────
# PASSO 2 – REDE NEURAL MULTICLASSE
# ───────────────────────────────────────────────────────────────────
# Arquitetura: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
#
# Forward:
#   z¹ = W¹·x + b¹    a¹ = ReLU(z¹)
#   z² = W²·a¹ + b²   a² = ReLU(z²)
#   z³ = W³·a² + b³   ŷ  = Softmax(z³)
#
# Loss: Categorical Cross-Entropy
#
# Backward:
#   δ³ = ŷ − y                          ← saída (softmax + CE)
#   dW³ = (a²)ᵀ · δ³ / m
#   δ² = (δ³ · (W³)ᵀ) ⊙ ReLU'(z²)
#   dW² = (a¹)ᵀ · δ² / m
#   δ¹ = (δ² · (W²)ᵀ) ⊙ ReLU'(z¹)
#   dW¹ = xᵀ · δ¹ / m

class MNISTClassifier:
    """MLP multiclasse: ReLU nas camadas ocultas, Softmax na saída."""

    def __init__(self, layer_sizes, learning_rate=0.1):
        self.lr         = learning_rate
        self.num_layers = len(layer_sizes) - 1
        self.train_losses = []; self.val_losses   = []
        self.train_accs   = []; self.val_accs     = []

        # He initialization para ReLU
        self.W = []
        self.b = []
        for i in range(self.num_layers):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            self.W.append(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            self.b.append(np.zeros((1, n_out)))

    def forward(self, X):
        self._z = []
        self._a = [X]
        a = X
        for l in range(self.num_layers - 1):    # camadas ocultas: ReLU
            z = np.dot(a, self.W[l]) + self.b[l]
            a = relu(z)
            self._z.append(z); self._a.append(a)
        # camada de saída: Softmax
        z = np.dot(a, self.W[-1]) + self.b[-1]
        a = softmax(z)
        self._z.append(z); self._a.append(a)
        return self._a[-1]

    def backward(self, y):
        m     = y.shape[0]
        delta = self._a[-1] - y   # δ = ŷ − y (softmax + cross-entropy)

        for l in reversed(range(self.num_layers)):
            dW = np.dot(self._a[l].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            self.W[l] -= self.lr * dW
            self.b[l]  -= self.lr * db
            if l > 0:
                delta = np.dot(delta, self.W[l].T) * relu_derivative(self._z[l - 1])

    def _train_epoch(self, X, y, batch_size):
        idx = np.random.permutation(X.shape[0])
        for s in range(0, X.shape[0], batch_size):
            e = min(s + batch_size, X.shape[0])
            self.forward(X[idx[s:e]])
            self.backward(y[idx[s:e]])

    def fit(self, X_tr, y_tr, X_val, y_val, epochs=50, batch_size=64, verbose=True):
        for epoch in range(epochs):
            self._train_epoch(X_tr, y_tr, batch_size)

            t_pred = self.forward(X_tr)
            v_pred = self.forward(X_val)
            t_loss = cross_entropy(y_tr,  t_pred)
            v_loss = cross_entropy(y_val, v_pred)
            t_acc  = np.mean(np.argmax(t_pred, axis=1) == np.argmax(y_tr,  axis=1))
            v_acc  = np.mean(np.argmax(v_pred, axis=1) == np.argmax(y_val, axis=1))

            self.train_losses.append(t_loss); self.val_losses.append(v_loss)
            self.train_accs.append(t_acc);    self.val_accs.append(v_acc)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Época {epoch+1:3d}/{epochs}  "
                      f"Train loss={t_loss:.4f} acc={t_acc*100:.1f}%  "
                      f"Val loss={v_loss:.4f} acc={v_acc*100:.1f}%")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y_oh):
        return np.mean(self.predict(X) == np.argmax(y_oh, axis=1))


# ───────────────────────────────────────────────────────────────────
# PASSO 3 – TREINAR E AVALIAR
# ───────────────────────────────────────────────────────────────────
print("\n2. Treinando MLP 784→128→64→10...")
np.random.seed(42)
modelo = MNISTClassifier([784, 128, 64, 10], learning_rate=0.1)
modelo.fit(X_train, y_train_oh, X_val, y_val_oh, epochs=50, batch_size=64, verbose=True)

test_acc = modelo.accuracy(X_test, y_test_oh)
y_pred   = modelo.predict(X_test)
print(f"\n  ✅ Acurácia no Test Set: {test_acc*100:.2f}%")

# ───────────────────────────────────────────────────────────────────
# PASSO 4 – GRÁFICOS E ANÁLISE DE ERROS
# ───────────────────────────────────────────────────────────────────

# Curvas de aprendizado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Parte 3 – Curvas de Aprendizado', fontsize=13, fontweight='bold')
axes[0].plot(modelo.train_losses, label='Treino', color='steelblue')
axes[0].plot(modelo.val_losses,   label='Validação', color='tomato')
axes[0].set_title('Loss (Cross-Entropy)'); axes[0].set_xlabel('Época')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in modelo.train_accs], label='Treino', color='steelblue')
axes[1].plot([a*100 for a in modelo.val_accs],   label='Validação', color='tomato')
axes[1].set_title('Acurácia (%)'); axes[1].set_xlabel('Época')
axes[1].set_ylim([50, 100]); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('GO0935-parte3-curvas.png', dpi=120, bbox_inches='tight')
plt.show()

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predito'); plt.ylabel('Verdadeiro')
plt.title(f'Matriz de Confusão  (Test Acc: {test_acc*100:.2f}%)')
plt.tight_layout()
plt.savefig('GO0935-parte3-confusao.png', dpi=120, bbox_inches='tight')
plt.show()

# Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=[str(d) for d in range(10)]))

# Erros mais confiantes
errors        = y_test != y_pred
error_idx     = np.where(errors)[0]
probs         = modelo.forward(X_test)
confidences   = np.max(probs, axis=1)
top20_idx     = error_idx[np.argsort(confidences[error_idx])[-20:]]

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
fig.suptitle('Top 20 Erros Mais Confiantes', fontsize=13, fontweight='bold')
for ax, idx in zip(axes.ravel(), top20_idx):
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'Real:{y_test[idx]} Pred:{y_pred[idx]}\n{confidences[idx]*100:.0f}%', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig('GO0935-parte3-erros.png', dpi=120, bbox_inches='tight')
plt.show()

# Acurácia por dígito
print("=" * 60)
print("ACURÁCIA POR DÍGITO")
print("=" * 60)
digit_accs = []
for d in range(10):
    m = y_test == d
    a = np.mean(y_pred[m] == y_test[m])
    digit_accs.append((d, a))
    barra = "█" * int(a * 20)
    print(f"  {d}: {barra:<20} {a*100:5.1f}%  ({m.sum()} amostras)")

easiest = max(digit_accs, key=lambda x: x[1])
hardest = min(digit_accs, key=lambda x: x[1])

# ───────────────────────────────────────────────────────────────────
# CHECKPOINT PARTE 3
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ CHECKPOINT PARTE 3 – RESUMO FINAL")
print("=" * 60)
print(f"\n📊 RESULTADOS:")
print(f"   Test Accuracy:      {test_acc*100:.2f}%")
print(f"   Erros:              {errors.sum()} / {len(y_test)}")
print(f"   Dígito mais fácil:  {easiest[0]} ({easiest[1]*100:.1f}%)")
print(f"   Dígito mais difícil:{hardest[0]} ({hardest[1]*100:.1f}%)")

print(f"\n🎯 NÍVEL:")
if test_acc >= 0.93:
    print("   🌟 AVANÇADO (≥93%) – Próximo passo: Batch Norm, Dropout")
elif test_acc >= 0.90:
    print("   ✅ INTERMEDIÁRIO (≥90%) – Tente lr=0.2 ou mais épocas")
elif test_acc >= 0.85:
    print("   ✅ MÍNIMO (≥85%) – Objetivo alcançado!")
else:
    print("   ⚠️ Abaixo de 85% – Verifique normalização e He init")

print(f"\n📚 CONCEITOS APLICADOS:")
for c in [
    "Neurônio artificial: z = W·x + b, a = f(z)",
    "ReLU nas camadas ocultas (evita vanishing gradient)",
    "Softmax na saída (distribuição de probabilidade)",
    "Categorical Cross-Entropy como função de custo",
    "He Initialization para estabilidade do treinamento",
    "Mini-batch Gradient Descent",
    "Backpropagation com Chain Rule",
    "Análise de overfitting (gap treino vs. validação)",
]:
    print(f"   ✓ {c}")

print("\n🎉🎉  ATIVIDADE 01 COMPLETA!  🎉🎉")
