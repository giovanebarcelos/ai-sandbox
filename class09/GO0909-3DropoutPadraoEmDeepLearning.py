# GO0909-3DropoutPadraoEmDeepLearning
# ═══════════════════════════════════════════════════════════════════
# DROPOUT — REGULARIZAÇÃO EM DEEP LEARNING
# Slide 25: Regularização em Redes Neurais
# ═══════════════════════════════════════════════════════════════════
"""
Dropout (Srivastava et al., 2014):
  - Durante treino: cada neurônio é desativado com probabilidade p
  - Máscara binária aleatória, escalada por 1/(1-p) (inverted dropout)
  - Durante inferência: todos neurônios ativos (sem dropout)
  - Efeito: evita co-adaptação, age como ensemble de sub-redes
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def dropout(a, keep_prob, training=True):
    """
    Inverted Dropout:
      - Cria máscara aleatória com probabilidade keep_prob de manter neurônio
      - Divide por keep_prob para manter esperança das ativações igual
    Args:
        a: ativações (ndarray)
        keep_prob: probabilidade de manter o neurônio (1 - taxa de dropout)
        training: bool — aplica dropout apenas durante treino
    Returns:
        a_dropout: ativações com dropout aplicado
        mask: máscara usada (para backprop)
    """
    if not training or keep_prob == 1.0:
        return a, np.ones_like(a)

    mask = (np.random.rand(*a.shape) < keep_prob) / keep_prob
    a_dropout = a * mask
    return a_dropout, mask


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLPDropout:
    """MLP com Dropout implementado do zero."""

    def __init__(self, sizes, keep_prob=0.8, lr=0.05, epochs=100, seed=42):
        np.random.seed(seed)
        self.keep_prob = keep_prob
        self.lr = lr
        self.epochs = epochs
        self.Ws = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
                   for i in range(len(sizes)-1)]
        self.bs = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    def _forward(self, X, training=True):
        a = X
        self._acts = [a]
        self._masks = []
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            if i < len(self.Ws) - 1:
                a = relu(z)
                a, mask = dropout(a, self.keep_prob, training=training)
            else:
                a = softmax(z)
                mask = np.ones_like(a)
            self._acts.append(a)
            self._masks.append(mask)
        return a

    def _bce_loss(self, probs, y_oh):
        return -np.mean(np.sum(y_oh * np.log(probs + 1e-9), axis=1))

    def fit(self, X_tr, y_tr, X_vl, y_vl):
        eye = np.eye(len(np.unique(y_tr)))
        for ep in range(self.epochs):
            probs = self._forward(X_tr, training=True)
            loss  = self._bce_loss(probs, eye[y_tr])

            dA = probs - eye[y_tr]
            for i in range(len(self.Ws)-1, -1, -1):
                dW = self._acts[i].T @ dA / len(X_tr)
                db = dA.mean(axis=0, keepdims=True)
                self.Ws[i] -= self.lr * dW
                self.bs[i] -= self.lr * db
                if i > 0:
                    dA = (dA @ self.Ws[i].T) * (self._acts[i] > 0) * self._masks[i-1]

            p_vl  = self._forward(X_vl, training=False)
            lv    = self._bce_loss(p_vl, eye[y_vl])
            acc   = np.mean(probs.argmax(1) == y_tr)
            acc_v = np.mean(p_vl.argmax(1) == y_vl)
            self.history["loss"].append(loss); self.history["val_loss"].append(lv)
            self.history["acc"].append(acc);   self.history["val_acc"].append(acc_v)


if __name__ == "__main__":
    # Dataset
    X, y = make_moons(n_samples=600, noise=0.25, random_state=42)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_vl = sc.transform(X_vl)

    sizes = [2, 128, 64, 2]

    print("Treinando SEM dropout...")
    m_nodrop = MLPDropout(sizes, keep_prob=1.0, lr=0.05, epochs=150)
    m_nodrop.fit(X_tr, y_tr, X_vl, y_vl)

    print("Treinando COM dropout (keep_prob=0.7)...")
    m_drop = MLPDropout(sizes, keep_prob=0.7, lr=0.05, epochs=150)
    m_drop.fit(X_tr, y_tr, X_vl, y_vl)

    acc_nd = np.mean(m_nodrop._forward(X_vl, training=False).argmax(1) == y_vl)
    acc_dr = np.mean(m_drop._forward(X_vl,  training=False).argmax(1) == y_vl)
    print(f"\n  Sem Dropout  — val acc: {acc_nd*100:.1f}%")
    print(f"  Com Dropout  — val acc: {acc_dr*100:.1f}%")

    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, hist, titulo in [
        (axes[0], m_nodrop.history, "Sem Dropout"),
        (axes[1], m_drop.history,  "Com Dropout (keep=0.7)")
    ]:
        ax.plot(hist["loss"],     label="Treino")
        ax.plot(hist["val_loss"], label="Validação", linestyle="--")
        ax.set_title(titulo)
        ax.set_xlabel("Época"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Efeito do Dropout no Overfitting", fontsize=13)
    plt.tight_layout()
    plt.savefig("GO0909_dropout.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0909_dropout.png")
    print("\n📌 Dropout reduz overfitting ao forçar a rede a aprender representações robustas")
