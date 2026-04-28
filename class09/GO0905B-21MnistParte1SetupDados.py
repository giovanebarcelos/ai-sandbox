# GO0905B-21MnistParte1SetupDados
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: RECONHECIMENTO DE DÍGITOS (MNIST)
# Etapa 1: SETUP E PREPARAÇÃO DOS DADOS
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ───────────────────────────────────────────────────────────────────
# 1. CARREGAR DATASET MNIST
# ───────────────────────────────────────────────────────────────────
print("1. Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"   Dataset carregado: X.shape={X.shape}, y.shape={y.shape}")

# ───────────────────────────────────────────────────────────────────
# 2. NORMALIZAÇÃO (pixel 0-255 → 0.0-1.0)
# ───────────────────────────────────────────────────────────────────
X = X / 255.0
print(f"   Normalização: min={X.min():.1f}, max={X.max():.1f}")

# ───────────────────────────────────────────────────────────────────
# 3. SPLIT: 6000 treino | 2000 validação | 2000 teste
# ───────────────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=6000, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"   Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

# ───────────────────────────────────────────────────────────────────
# 4. ONE-HOT ENCODING
# ───────────────────────────────────────────────────────────────────
def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]

y_train_oh = one_hot_encode(y_train)
y_val_oh   = one_hot_encode(y_val)
y_test_oh  = one_hot_encode(y_test)
print(f"   One-hot: y_train_oh.shape={y_train_oh.shape}")

# ───────────────────────────────────────────────────────────────────
# 5. VISUALIZAR AMOSTRAS
# ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Amostras do MNIST - Treino', fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

if __name__ == "__main__":
    # ───────────────────────────────────────────────────────────────
    # ✅ CHECKPOINT ETAPA 1:
    # ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("✅ CHECKPOINT ETAPA 1 - VALIDAÇÕES")
    print("="*70)
    assert X.shape == (70000, 784),          f"ERRO: X.shape={X.shape}, esperado (70000, 784)"
    assert X.min() >= 0.0 and X.max() <= 1.0, "ERRO: normalização fora de [0, 1]"
    assert X_train.shape[0] == 6000,         f"ERRO: train={X_train.shape[0]}, esperado 6000"
    assert X_val.shape[0]   == 2000,         f"ERRO: val={X_val.shape[0]}, esperado 2000"
    assert X_test.shape[0]  == 2000,         f"ERRO: test={X_test.shape[0]}, esperado 2000"
    assert y_train_oh.shape == (6000, 10),   f"ERRO: one-hot shape={y_train_oh.shape}"

    print(f"✓ Dataset: X.shape={X.shape}")
    print(f"✓ Normalização: [{X.min():.1f}, {X.max():.1f}]")
    print(f"✓ Split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    print(f"✓ One-hot: y_train_oh.shape={y_train_oh.shape}")
    print("\n🎉 Etapa 1 completa! Prossiga para GO0906 (Etapa 2 - Arquitetura)")
