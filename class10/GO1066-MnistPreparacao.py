# GO1066-MnistPreparacao
# Projeto MNIST - Parte 1: carrega o dataset, inspeciona shapes e distribuição
# das classes, normaliza pixels e reshape para entrada de rede Dense.
from tensorflow import keras
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# 1) Carregar MNIST
(X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.mnist.load_data()

print("=== ANÁLISE DO DATASET ===")
print(f"X_train shape : {X_train_raw.shape}")   # (60000, 28, 28)
print(f"X_test  shape : {X_test_raw.shape}")    # (10000, 28, 28)
print(f"y_train shape : {y_train.shape}")       # (60000,)
print(f"Pixel min/max : {X_train_raw.min()} / {X_train_raw.max()}")  # 0 / 255
print(f"Classes       : {np.unique(y_train)}")  # 0..9

# 2) Distribuição das classes (verificar balanceamento)
print("\n=== DISTRIBUIÇÃO POR CLASSE ===")
for classe in range(10):
    qtd = np.sum(y_train == classe)
    print(f"  Dígito {classe}: {qtd} amostras ({qtd/len(y_train)*100:.1f}%)")

# 3) Visualizar amostras
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_raw[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")
plt.suptitle("Amostras MNIST (antes da normalização)")
plt.tight_layout()
plt.show()

# 4) Normalização: 0-255 → 0-1 (pixels controlados, gradientes estáveis)
X_train = X_train_raw.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test_raw.reshape(-1, 784).astype("float32") / 255.0

print("\n=== APÓS PRÉ-PROCESSAMENTO ===")
print(f"X_train shape : {X_train.shape}")   # (60000, 784)
print(f"X_test  shape : {X_test.shape}")    # (10000, 784)
print(f"Pixel min/max : {X_train.min():.2f} / {X_train.max():.2f}")  # 0.0 / 1.0

# 5) Comparação visual: histograma antes vs depois da normalização
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(X_train_raw.flatten(), bins=50, color="steelblue")
axes[0].set_title("Distribuição dos pixels ANTES da normalização")
axes[0].set_xlabel("Valor do pixel (0-255)")
axes[0].set_ylabel("Frequência")

axes[1].hist(X_train.flatten(), bins=50, color="darkorange")
axes[1].set_title("Distribuição dos pixels APÓS normalização")
axes[1].set_xlabel("Valor do pixel (0.0-1.0)")
axes[1].set_ylabel("Frequência")

plt.tight_layout()
plt.show()

print("\nDados prontos para a Parte 2 (definição da arquitetura).")
