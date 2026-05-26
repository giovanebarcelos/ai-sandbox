# GO0917-ReduzirComplexidade
# ═══════════════════════════════════════════════════════════════════
# ESTRATÉGIAS PARA REDUZIR OVERFITTING — REDUZIR COMPLEXIDADE
# Slide 25: Regularização em Redes Neurais
# ═══════════════════════════════════════════════════════════════════
"""
Quando a rede está fazendo overfitting (treino >> validação),
três estratégias imediatas: reduzir modelo, treinar menos, regularizar.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def relu(z):      return np.maximum(0, z)
def sigmoid(z):   return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def treinar_mlp(X_tr, y_tr, X_val, y_val,
                hidden_sizes, epochs, lr=0.05, seed=42):
    """MLP binário simples para comparar complexidades."""
    np.random.seed(seed)
    sizes = [X_tr.shape[1]] + list(hidden_sizes) + [1]
    Ws = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
          for i in range(len(sizes)-1)]
    bs = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]

    hist_tr, hist_vl = [], []

    for _ in range(epochs):
        # Forward
        a = X_tr
        acts = [a]
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = a @ W + b
            a = relu(z) if i < len(Ws)-1 else sigmoid(z)
            acts.append(a)

        # Loss BCE
        p = acts[-1].ravel()
        loss = -np.mean(y_tr * np.log(p+1e-9) + (1-y_tr)*np.log(1-p+1e-9))

        # Backward
        dA = (acts[-1].ravel() - y_tr).reshape(-1, 1)
        for i in range(len(Ws)-1, -1, -1):
            dW = acts[i].T @ dA / len(X_tr)
            db = dA.mean(axis=0, keepdims=True)
            Ws[i] -= lr * dW
            bs[i] -= lr * db
            if i > 0:
                dA = (dA @ Ws[i].T) * (acts[i] > 0)

        # Validação
        av = X_val
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = av @ W + b
            av = relu(z) if i < len(Ws)-1 else sigmoid(z)
        pv = av.ravel()
        loss_v = -np.mean(y_val*np.log(pv+1e-9) + (1-y_val)*np.log(1-pv+1e-9))
        hist_tr.append(loss)
        hist_vl.append(loss_v)

    acc_tr = np.mean((acts[-1].ravel() > 0.5) == y_tr)
    acc_vl = np.mean((pv > 0.5) == y_val)
    return hist_tr, hist_vl, acc_tr, acc_vl


if __name__ == '__main__':
    # Dataset
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.3, random_state=42)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr); X_vl = sc.transform(X_vl)

    # ─── Estratégia 1: Reduzir complexidade do modelo ─────────────
    print("=" * 60)
    print("ESTRATÉGIA 1 — Reduzir complexidade do modelo")
    print("=" * 60)
    # Modelo complexo (pode overfittar)
    hidden_grande = [256, 128, 64]   # 3 camadas grandes
    # Modelo simples (menos parâmetros)
    hidden_pequeno = [64, 32]        # 2 camadas menores

    loss_g_tr, loss_g_vl, acc_g_tr, acc_g_vl = treinar_mlp(
        X_tr, y_tr, X_vl, y_vl, hidden_grande, epochs=200)
    loss_p_tr, loss_p_vl, acc_p_tr, acc_p_vl = treinar_mlp(
        X_tr, y_tr, X_vl, y_vl, hidden_pequeno, epochs=200)

    print(f"  Modelo grande  {hidden_grande}: "
          f"acc_tr={acc_g_tr*100:.1f}%  acc_val={acc_g_vl*100:.1f}%")
    print(f"  Modelo pequeno {hidden_pequeno}: "
          f"acc_tr={acc_p_tr*100:.1f}%  acc_val={acc_p_vl*100:.1f}%")
    gap_g = (acc_g_tr - acc_g_vl)*100
    gap_p = (acc_p_tr - acc_p_vl)*100
    print(f"  Gap treino-val (grande):  {gap_g:.1f}%")
    print(f"  Gap treino-val (pequeno): {gap_p:.1f}%  ← menor overfitting")

    # ─── Estratégia 2: Parar treinamento mais cedo ────────────────
    print("\n" + "=" * 60)
    print("ESTRATÉGIA 2 — Parar treinamento mais cedo (Early Stopping)")
    print("=" * 60)
    epochs_longo  = 300   # pode overfittar
    epochs_curto  = 50    # para mais cedo

    loss_l_tr, loss_l_vl, acc_l_tr, acc_l_vl = treinar_mlp(
        X_tr, y_tr, X_vl, y_vl, [128, 64], epochs=epochs_longo)
    loss_c_tr, loss_c_vl, acc_c_tr, acc_c_vl = treinar_mlp(
        X_tr, y_tr, X_vl, y_vl, [128, 64], epochs=epochs_curto)

    print(f"  {epochs_longo} épocas: acc_tr={acc_l_tr*100:.1f}% "
          f" acc_val={acc_l_vl*100:.1f}%")
    print(f"  {epochs_curto} épocas: acc_tr={acc_c_tr*100:.1f}% "
          f" acc_val={acc_c_vl*100:.1f}%")

    # Encontrar melhor época (val loss mínima)
    best_epoch = int(np.argmin(loss_l_vl)) + 1
    print(f"  Melhor época pelo val_loss: {best_epoch}")

    # ─── Gráficos comparativos ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(loss_g_tr, label='Grande - Treino',  color='red',  alpha=0.7)
    axes[0].plot(loss_g_vl, label='Grande - Val',     color='red',  linestyle='--')
    axes[0].plot(loss_p_tr, label='Pequeno - Treino', color='blue', alpha=0.7)
    axes[0].plot(loss_p_vl, label='Pequeno - Val',    color='blue', linestyle='--')
    axes[0].set_title('Estratégia 1: Reduzir Complexidade')
    axes[0].set_xlabel('Época'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss_l_tr, label='Treino',    color='blue')
    axes[1].plot(loss_l_vl, label='Validação', color='orange')
    axes[1].axvline(best_epoch-1, color='green', linestyle='--',
                    label=f'Melhor época ({best_epoch})')
    axes[1].set_title('Estratégia 2: Early Stopping')
    axes[1].set_xlabel('Época'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Estratégias para Reduzir Overfitting', fontsize=13)
    plt.tight_layout()
    plt.savefig('GO0917_reduzir_complexidade.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("\nSalvo: GO0917_reduzir_complexidade.png")

    print("\n📌 RESUMO:")
    print("  1. Reduzir hidden_sizes → menos parâmetros → menos overfitting")
    print("  2. Menos épocas / Early Stopping → para antes de overfittar")
    print("  3. Dropout / L2 Regularização → próximo: GO0909")
