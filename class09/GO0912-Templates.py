# GO0912-Templates
# ═══════════════════════════════════════════════════════════════════
# TEMPLATE — CARREGAR E PREPARAR MNIST
# Slide 21A: Setup e Dados
# ═══════════════════════════════════════════════════════════════════
"""
Template de referência para carregar e preparar o dataset MNIST.
Pode ser usado com Keras (mnist.load_data) ou scikit-learn (fetch_openml).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TEMPLATE_MNIST_KERAS = """
# ── Template MNIST com Keras ─────────────────────────────
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_tr, y_tr), (X_te, y_te) = mnist.load_data()

# Flatten 28×28 → 784 e normalizar 0-255 → 0.0-1.0
X_tr = X_tr.reshape(-1, 784) / 255.0
X_te = X_te.reshape(-1, 784) / 255.0

# One-hot encoding
y_tr_oh = to_categorical(y_tr, 10)
y_te_oh = to_categorical(y_te, 10)

print(f"X_tr: {X_tr.shape}  y_tr_oh: {y_tr_oh.shape}")
# ─────────────────────────────────────────────────────────
"""

TEMPLATE_MNIST_SKLEARN = """
# ── Template MNIST com scikit-learn ──────────────────────
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=6000, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
# ─────────────────────────────────────────────────────────
"""


if __name__ == "__main__":
    print("=" * 60)
    print("TEMPLATE — CARREGAR E PREPARAR MNIST")
    print("=" * 60)
    print("\n[Keras]"); print(TEMPLATE_MNIST_KERAS)
    print("[scikit-learn]"); print(TEMPLATE_MNIST_SKLEARN)

    # Demonstração executável com subset do MNIST
    print("\nCarregando MNIST (subconjunto 2000 amostras)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data[:2000], mnist.target[:2000].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=1400, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print(f"  Faixa X_train: [{X_train.min():.2f}, {X_train.max():.2f}]")

    # Visualizar exemplos
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.ravel()):
        img = scaler.inverse_transform(X_train[i].reshape(1,-1)).reshape(28,28)
        ax.imshow(img, cmap="gray"); ax.set_title(f"Label: {y_train[i]}")
        ax.axis("off")
    plt.suptitle("Exemplos do Dataset MNIST", fontsize=12)
    plt.tight_layout()
    plt.savefig("GO0912_mnist_exemplos.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0912_mnist_exemplos.png")
    print("\n✅ Dataset MNIST carregado e preparado com sucesso!")
