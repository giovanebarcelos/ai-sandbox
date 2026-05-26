# GO0930-MLP
# ═══════════════════════════════════════════════════════════════════
# TEMPLATE MLP DO ZERO — ESTRUTURA COMPLETA
# Slide 29: Implementação Completa do MLP
# ═══════════════════════════════════════════════════════════════════
"""
Estrutura completa de um MLP implementado do zero com NumPy.
Inclui: forward, backward, fit, predict, evaluate.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class MLP:
    """
    Multi-Layer Perceptron (MLP) implementado do zero.
    Suporta classificação binária e multiclasse.
    """

    def __init__(self, hidden_sizes=(128, 64), lr=0.05,
                 epochs=100, batch_size=32, seed=42):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        np.random.seed(seed)
        self.Ws = []
        self.bs = []
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}

    # ── Funções auxiliares ────────────────────────────────────────

    def _relu(self, z):
        return np.maximum(0, z)

    def _softmax(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _cross_entropy(self, probs, y_oh):
        return -np.mean(np.sum(y_oh * np.log(probs + 1e-9), axis=1))

    def _init_weights(self, n_in, n_classes):
        sizes = [n_in] + list(self.hidden_sizes) + [n_classes]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.Ws.append(W)
            self.bs.append(b)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, X):
        self._acts = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            a = self._relu(z) if i < len(self.Ws) - 1 else self._softmax(z)
            self._acts.append(a)
        return a

    # ── Backward ──────────────────────────────────────────────────

    def _backward(self, X, y_oh):
        probs = self.forward(X)
        dA = probs - y_oh
        for i in range(len(self.Ws) - 1, -1, -1):
            dW = self._acts[i].T @ dA / len(X)
            db = dA.mean(axis=0, keepdims=True)
            self.Ws[i] -= self.lr * dW
            self.bs[i] -= self.lr * db
            if i > 0:
                dA = (dA @ self.Ws[i].T) * (self._acts[i] > 0)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        n_classes = len(np.unique(y_train))
        self._init_weights(X_train.shape[1], n_classes)
        eye = np.eye(n_classes)

        for ep in range(self.epochs):
            # Mini-batch shuffle
            idx = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), self.batch_size):
                bi = idx[start:start+self.batch_size]
                self._backward(X_train[bi], eye[y_train[bi]])

            # Métricas por época
            p_tr = self.forward(X_train)
            loss = self._cross_entropy(p_tr, eye[y_train])
            acc  = np.mean(p_tr.argmax(1) == y_train)
            self.history["loss"].append(loss)
            self.history["acc"].append(acc)

            if X_val is not None:
                p_vl = self.forward(X_val)
                lv   = self._cross_entropy(p_vl, eye[y_val])
                av   = np.mean(p_vl.argmax(1) == y_val)
                self.history["val_loss"].append(lv)
                self.history["val_acc"].append(av)
            else:
                self.history["val_loss"].append(None)
                self.history["val_acc"].append(None)

            if (ep + 1) % 20 == 0:
                val_info = f"  val_loss={lv:.4f}" if X_val is not None else ""
                print(f"  Ep {ep+1:3d} | loss={loss:.4f} | acc={acc*100:.1f}%{val_info}")

    # ── Predict / Evaluate ────────────────────────────────────────

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))


# ── Todas as funções auxiliares (implementadas acima na classe) ───
def implementar_todas_funcoes_auxiliares():
    """Checklist de funções necessárias para um MLP completo."""
    funcoes = {
        "__init__": "Inicializa pesos (He init), hiperparâmetros, histórico",
        "forward":  "Propaga entrada → saída (ReLU + Softmax)",
        "backward": "Calcula gradientes e atualiza pesos (backprop)",
        "fit":      "Loop de treinamento com mini-batches",
        "predict":  "Inferência (argmax da saída softmax)",
        "evaluate": "Calcula acurácia em um conjunto",
        "_relu":    "Função de ativação ReLU",
        "_softmax": "Função de saída para classificação multiclasse",
        "_cross_entropy": "Função de perda (categorical cross-entropy)",
    }
    for fn, desc in funcoes.items():
        print(f"  {fn:20s}: {desc}")


if __name__ == "__main__":
    print("=" * 60)
    print("MLP — ESTRUTURA COMPLETA E FUNÇÕES AUXILIARES")
    print("=" * 60)
    implementar_todas_funcoes_auxiliares()

    # Treinar no make_moons
    print("\nTreinando MLP em make_moons...")
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

    mlp = MLP(hidden_sizes=(64, 32), lr=0.05, epochs=80, batch_size=32)
    mlp.fit(X_tr, y_tr, X_te, y_te)
    acc = mlp.evaluate(X_te, y_te)
    print(f"\nTest Accuracy: {acc*100:.1f}%")

    plt.figure(figsize=(8, 3))
    plt.plot(mlp.history["loss"],     label="Treino")
    plt.plot(mlp.history["val_loss"], label="Validação", linestyle="--")
    plt.xlabel("Época"); plt.ylabel("Loss")
    plt.title("MLP do zero — make_moons"); plt.legend()
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("GO0930_mlp.png", dpi=100, bbox_inches="tight"); plt.show()
    print("Salvo: GO0930_mlp.png")
