# GO0913-Go0912templates
# ═══════════════════════════════════════════════════════════════════
# TEMPLATE — NORMALIZAÇÃO COM STANDARDSCALER
# Slide 21A: Pré-processamento de Dados
# ═══════════════════════════════════════════════════════════════════
"""
StandardScaler: normaliza cada feature para média 0 e desvio padrão 1.
  z = (x - media) / desvio_padrao

Regra fundamental: fit apenas no treino, transform em treino+val+teste.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    print("=" * 60)
    print("STANDARDSCALER — NORMALIZAÇÃO DE DADOS")
    print("=" * 60)

    # Carregar subset do MNIST
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data[:3000], mnist.target[:3000].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=2000, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # ── Normalização ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit + transform no treino
    X_val   = scaler.transform(X_val)          # só transform (sem fit!)
    X_test  = scaler.transform(X_test)         # só transform (sem fit!)

    print(f"  Antes: min={0:.0f}, max={255:.0f}")
    print(f"  Depois X_train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"  Depois X_val:   mean={X_val.mean():.4f},   std={X_val.std():.4f}")
    print(f"  Depois X_test:  mean={X_test.mean():.4f},  std={X_test.std():.4f}")

    # ── Por que não fazer fit no val/test? ──────────────────────
    print("\n📌 IMPORTANTE: Regra do pré-processamento")
    print("  scaler.fit_transform(X_train)  ← aprende média/std do treino")
    print("  scaler.transform(X_val)        ← usa média/std do TREINO")
    print("  scaler.transform(X_test)       ← usa média/std do TREINO")
    print("  → Evita data leakage (vazamento de info do futuro)")

    # ── Visualização antes/depois ────────────────────────────────
    mnist_raw = fetch_openml("mnist_784", version=1, as_frame=False)
    X_raw = mnist_raw.data[:2000]
    X_raw_scaled = StandardScaler().fit_transform(X_raw)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pixel_idx = 300  # pixel de exemplo
    axes[0].hist(X_raw[:, pixel_idx], bins=30, color="salmon", edgecolor="white")
    axes[0].set_title(f"Pixel {pixel_idx} — Antes da normalização")
    axes[0].set_xlabel("Intensidade (0-255)")

    axes[1].hist(X_raw_scaled[:, pixel_idx], bins=30, color="steelblue", edgecolor="white")
    axes[1].set_title(f"Pixel {pixel_idx} — Após StandardScaler")
    axes[1].set_xlabel("Valor normalizado (z-score)")

    plt.tight_layout()
    plt.savefig("GO0913_standardscaler.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("\nSalvo: GO0913_standardscaler.png")
    print("\n✅ Normalização aplicada corretamente!")
