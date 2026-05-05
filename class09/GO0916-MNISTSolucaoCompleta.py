# GO0916-MNISTSolucaoCompleta
# ═══════════════════════════════════════════════════════════════════
# SOLUÇÃO COMPLETA - ATIVIDADE PRÁTICA MNIST (Slides 21A → 24)
#   Parte 1: Setup e Dados            (Slide 21A / GO0905B)
#   Parte 2: Arquitetura da Rede      (Slide 22  / GO0906)
#   Parte 3: Treinamento              (Slide 23  / GO0907)
#   Parte 4: Análise de Erros         (Slide 24  / GO0908)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ───────────────────────────────────────────────────────────────────
# PARTE 1 – SETUP E DADOS  (Slide 21A)
# ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("PARTE 1 – SETUP E DADOS")
print("=" * 70)

print("1. Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"   Dataset carregado: X.shape={X.shape}, y.shape={y.shape}")

# Normalização (pixel 0-255 → 0.0-1.0)
X = X / 255.0
print(f"   Normalização: min={X.min():.1f}, max={X.max():.1f}")

# Split: 6000 treino | 2000 validação | 2000 teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=6000, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"   Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# One-hot encoding
def one_hot_encode(labels, n_classes=10):
    return np.eye(n_classes)[labels]

y_train_oh = one_hot_encode(y_train)
y_val_oh   = one_hot_encode(y_val)
y_test_oh  = one_hot_encode(y_test)
print(f"   One-hot: y_train_oh.shape={y_train_oh.shape}")

# Visualizar 10 amostras
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Parte 1 – Amostras MNIST (treino)', fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig("GO0916-parte1-amostras.png", dpi=120, bbox_inches='tight')
plt.show()

# Checkpoint Parte 1
assert X.shape == (70000, 784)
assert 0.0 <= X.min() and X.max() <= 1.0
assert X_train.shape[0] == 6000
assert X_val.shape[0] == 2000
assert X_test.shape[0] == 2000
assert y_train_oh.shape == (6000, 10)
print("\n✅ CHECKPOINT PARTE 1 – OK")

# ───────────────────────────────────────────────────────────────────
# PARTE 2 – ARQUITETURA DA REDE  (Slide 22)
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PARTE 2 – ARQUITETURA DA REDE")
print("=" * 70)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class MulticlassNN:
    """Rede neural multiclasse: ReLU nas ocultas, Softmax na saída."""

    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.lr = learning_rate
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1

        # He initialization para ReLU
        self.weights = []
        self.biases  = []
        for i in range(self.num_layers):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            self.weights.append(np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in))
            self.biases.append(np.zeros((1, n_out)))

        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []

    def forward(self, X):
        self.z_values = []
        self.a_values = [X]
        for i in range(self.num_layers - 1):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.a_values.append(relu(z))
        # Camada de saída com Softmax
        z = np.dot(self.a_values[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.a_values.append(softmax(z))
        return self.a_values[-1]

    def backward(self, X, y):
        m = X.shape[0]
        delta = self.a_values[-1] - y   # gradiente softmax + cross-entropy
        for i in reversed(range(self.num_layers)):
            dW = np.dot(self.a_values[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.z_values[i - 1])

    def _train_epoch(self, X, y, batch_size):
        idx = np.random.permutation(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            b   = idx[start:end]
            self.forward(X[b])
            self.backward(X[b], y[b])

    def fit(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=True):
        for epoch in range(epochs):
            self._train_epoch(X_train, y_train, batch_size)
            # Métricas
            t_pred = self.forward(X_train)
            v_pred = self.forward(X_val)
            t_loss = categorical_cross_entropy(y_train, t_pred)
            v_loss = categorical_cross_entropy(y_val,   v_pred)
            t_acc  = self.accuracy(X_train, y_train)
            v_acc  = self.accuracy(X_val,   y_val)
            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)
            self.train_accs.append(t_acc)
            self.val_accs.append(v_acc)
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Época {epoch+1:3d}/{epochs}  "
                      f"Train loss={t_loss:.4f} acc={t_acc*100:.1f}%  "
                      f"Val loss={v_loss:.4f} acc={v_acc*100:.1f}%")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))


print("   Input:    784 neurônios (28×28 pixels)")
print("   Hidden 1: 128 neurônios (ReLU)")
print("   Hidden 2:  64 neurônios (ReLU)")
print("   Output:    10 neurônios (Softmax)")

# Checkpoint Parte 2
_test_nn  = MulticlassNN(784, [128, 64], 10)
_test_out = _test_nn.forward(X_train[:10])
assert _test_out.shape == (10, 10)
assert abs(_test_out[0].sum() - 1.0) < 1e-6
print(f"\n✅ CHECKPOINT PARTE 2 – OK  "
      f"(parâmetros: {sum(w.size for w in _test_nn.weights) + sum(b.size for b in _test_nn.biases):,})")

# ───────────────────────────────────────────────────────────────────
# PARTE 3 – TREINAMENTO  (Slide 23)
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PARTE 3 – TREINAMENTO")
print("=" * 70)

nn = MulticlassNN(
    input_size=784,
    hidden_sizes=[128, 64],
    output_size=10,
    learning_rate=0.1
)

nn.fit(X_train, y_train_oh, X_val, y_val_oh, epochs=50, batch_size=64, verbose=True)

test_acc  = nn.accuracy(X_test, y_test_oh)
y_pred    = nn.predict(X_test)
print(f"\n✅ Acurácia final no Test Set: {test_acc*100:.2f}%")

# Curvas de aprendizado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Parte 3 – Curvas de Aprendizado', fontsize=13, fontweight='bold')

axes[0].plot(nn.train_losses, label='Treino', color='steelblue')
axes[0].plot(nn.val_losses,   label='Validação', color='tomato')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss (cross-entropy)')
axes[0].set_title('Curva de Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([a * 100 for a in nn.train_accs], label='Treino', color='steelblue')
axes[1].plot([a * 100 for a in nn.val_accs],   label='Validação', color='tomato')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Acurácia (%)')
axes[1].set_title('Curva de Acurácia')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([50, 100])

plt.tight_layout()
plt.savefig("GO0916-parte3-treinamento.png", dpi=120, bbox_inches='tight')
plt.show()

# Checkpoint Parte 3
gap = nn.train_accs[-1] - nn.val_accs[-1]
print(f"\n✅ CHECKPOINT PARTE 3 – OK")
print(f"   Train acc: {nn.train_accs[-1]*100:.2f}%  "
      f"Val acc: {nn.val_accs[-1]*100:.2f}%  "
      f"Gap: {gap*100:.2f}% {'⚠️ Overfitting' if gap > 0.15 else '✅ OK'}")

# ───────────────────────────────────────────────────────────────────
# PARTE 4 – ANÁLISE DE ERROS  (Slide 24)
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PARTE 4 – ANÁLISE DE ERROS")
print("=" * 70)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title(f'Parte 4 – Matriz de Confusão  (Acc: {test_acc*100:.2f}%)')
plt.tight_layout()
plt.savefig("GO0916-parte4-confusao.png", dpi=120, bbox_inches='tight')
plt.show()

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=[str(d) for d in range(10)]))

# Erros mais confiantes (rede errou com alta confiança)
print("=" * 70)
print("TOP 20 ERROS MAIS CONFIANTES")
print("=" * 70)
errors          = y_test != y_pred
error_indices   = np.where(errors)[0]
probs           = nn.forward(X_test)
confidences     = np.max(probs, axis=1)
top_errors_idx  = error_indices[np.argsort(confidences[error_indices])[-20:]]

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
fig.suptitle('Top 20 Erros Mais Confiantes', fontsize=13, fontweight='bold')
for ax, idx in zip(axes.ravel(), top_errors_idx):
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'Real: {y_test[idx]}  Pred: {y_pred[idx]}\n'
                 f'Conf: {confidences[idx]*100:.1f}%', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig("GO0916-parte4-erros.png", dpi=120, bbox_inches='tight')
plt.show()

# Acurácia por dígito
print("=" * 70)
print("ACURÁCIA POR DÍGITO")
print("=" * 70)
digit_accs = []
for d in range(10):
    mask    = (y_test == d)
    acc     = np.mean(y_pred[mask] == y_test[mask])
    digit_accs.append((d, acc))
    print(f"  Dígito {d}: {acc*100:5.2f}%  ({mask.sum()} amostras)")

# Top 10 confusões mais comuns
print("\n" + "=" * 70)
print("TOP 10 CONFUSÕES MAIS COMUNS")
print("=" * 70)
confusions = sorted(
    [(cm[i, j], i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0],
    reverse=True
)
print(f"{'Count':>5} | {'Real':>4} | {'Pred':>4} | Descrição")
print("-" * 45)
for count, true_d, pred_d in confusions[:10]:
    print(f"  {count:4d} |    {true_d} |    {pred_d} | {true_d} confundido com {pred_d}")

# ───────────────────────────────────────────────────────────────────
# CHECKPOINT FINAL
# ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅ CHECKPOINT FINAL – RESUMO DA ATIVIDADE COMPLETA")
print("=" * 70)

easiest = max(digit_accs, key=lambda x: x[1])
hardest = min(digit_accs, key=lambda x: x[1])

print(f"\n📊 RESULTADOS FINAIS:")
print(f"   Test Accuracy:         {test_acc*100:.2f}%")
print(f"   Total de erros:        {errors.sum()} / {len(y_test)} ({errors.sum()/len(y_test)*100:.1f}%)")
print(f"   Dígito mais fácil:     {easiest[0]} ({easiest[1]*100:.1f}%)")
print(f"   Dígito mais difícil:   {hardest[0]} ({hardest[1]*100:.1f}%)")

print(f"\n🎯 NÍVEL ALCANÇADO:")
if test_acc >= 0.93:
    print("   🌟 AVANÇADO (≥93%)  – Sugestões: Batch Norm, Early Stopping, dataset completo")
elif test_acc >= 0.90:
    print("   ✅ INTERMEDIÁRIO (≥90%) – Sugestões: lr=0.2, mais neurônios")
elif test_acc >= 0.85:
    print("   ✅ MÍNIMO (≥85%) – Sugestões: 100 épocas, arquitetura [256, 128]")
else:
    print("   ⚠️ ABAIXO DO MÍNIMO – Verifique normalização, lr e He initialization")

print(f"\n📚 CONCEITOS APLICADOS NESTA ATIVIDADE:")
for c in [
    "Forward propagation com múltiplas camadas",
    "Backpropagation com chain rule",
    "Mini-batch gradient descent",
    "ReLU (camadas ocultas) + Softmax (saída)",
    "Categorical cross-entropy loss",
    "He initialization para estabilidade",
    "Análise de overfitting (treino vs. validação)",
    "Matriz de confusão e classification report",
    "Análise de erros mais confiantes",
]:
    print(f"   ✓ {c}")

print("\n🎉🎉🎉  ATIVIDADE COMPLETA – PARABÉNS!  🎉🎉🎉")
