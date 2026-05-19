# GO1114-TreinamentoHibrido
# ANFIS (Adaptive Neuro-Fuzzy Inference System) — combina redes neurais
# e lógica fuzzy: os parâmetros das MFs são ajustados automaticamente pelo
# algoritmo híbrido de Jang (mínimos quadrados + backpropagation).
# Útil quando não se tem conhecimento especialista para definir as MFs manualmente.
#
# Exemplo: aproximação da função y = sin(x1) * cos(x2) com 2 entradas.
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Instalação automática de dependências ──────────────────────────────
import subprocess, sys, importlib

def _pip_install(package):
    """Instala pacote via pip silenciosamente."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", package],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

for _pkg in ["anfis", "numpy", "matplotlib"]:
    try:
        importlib.import_module(_pkg)
    except ModuleNotFoundError:
        print(f"Instalando {_pkg}...", end=" ", flush=True)
        _pip_install(_pkg)
        print("OK")

# ── 2. Imports ────────────────────────────────────────────────────────────
from anfis import ANFIS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 3. Dados sintéticos: y = sin(x1) * cos(x2) + ruído ───────────────────
np.random.seed(42)
N = 200
X = np.random.uniform(-np.pi, np.pi, (N, 2))
y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + np.random.normal(0, 0.05, N)

split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Dataset: {N} amostras  |  Treino: {split}  |  Teste: {N - split}")

# ── 4. Funções de pertinência (3 gaussianas por feature) ──────────────────
mf = [
    [['gaussmf', {'mean': -2.0, 'sigma': 1.5}],
     ['gaussmf', {'mean':  0.0, 'sigma': 1.5}],
     ['gaussmf', {'mean':  2.0, 'sigma': 1.5}]],
    [['gaussmf', {'mean': -2.0, 'sigma': 1.5}],
     ['gaussmf', {'mean':  0.0, 'sigma': 1.5}],
     ['gaussmf', {'mean':  2.0, 'sigma': 1.5}]]
]

# ── 5. Visualizar MFs iniciais ────────────────────────────────────────────
def plot_mfs(mf_params, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"ANFIS — Funções de Pertinência {title_suffix}", fontsize=13, fontweight='bold')
    x_range = np.linspace(-np.pi, np.pi, 300)
    colors = ["steelblue", "tomato", "seagreen"]
    for ax, feat_idx, feat_name in zip(axes, [0, 1], ["Entrada x₁", "Entrada x₂"]):
        for i, (_, params) in enumerate(mf_params[feat_idx]):
            mu = np.exp(-0.5 * ((x_range - params['mean']) / params['sigma']) ** 2)
            ax.plot(x_range, mu, color=colors[i % len(colors)],
                    label=f"MF {i+1}  μ={params['mean']:.2f}", linewidth=2)
        ax.set_title(feat_name)
        ax.set_xlabel("Valor de entrada")
        ax.set_ylabel("Grau de pertinência μ")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_mfs(mf, title_suffix="— Antes do Treino")

# ── 6. Criar e treinar ANFIS ──────────────────────────────────────────────
print("\nTreinando ANFIS (algoritmo híbrido de Jang)...")
anf = ANFIS(X_train, y_train, mf)
anf.trainHybridJangOffLine(epochs=50)
print("Treinamento concluído!")

# ── 7. Curva de erro durante o treino ─────────────────────────────────────
if hasattr(anf, 'errors') and len(anf.errors) > 0:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(anf.errors, color="steelblue", linewidth=2)
    ax.set_title("ANFIS — Curva de Erro por Época", fontsize=12, fontweight='bold')
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ── 8. Previsão no conjunto de teste ─────────────────────────────────────
y_pred = anf.predict(X_test)

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mae  = np.mean(np.abs(y_test - y_pred))
r2   = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

print(f"\nMetricas no conjunto de teste:")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")

# ── 9. Visualização dos resultados ────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("ANFIS — Resultados de Treinamento Híbrido", fontsize=14, fontweight='bold')

# 9a. Predição vs Real (linha)
ax1 = fig.add_subplot(gs[0, :])
idx = np.argsort(y_test)
ax1.plot(y_test[idx],  color="steelblue", linewidth=1.5, label="Real")
ax1.plot(y_pred[idx],  color="tomato",    linewidth=1.5, linestyle="--", label="Predito (ANFIS)")
ax1.set_title("Predição vs Real (amostras de teste ordenadas por y_real)")
ax1.set_xlabel("Amostra")
ax1.set_ylabel("y")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 9b. Scatter: predito vs real
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", s=40)
lims = [min(y_test.min(), y_pred.min()) - 0.1, max(y_test.max(), y_pred.max()) + 0.1]
ax2.plot(lims, lims, "r--", linewidth=1.5, label="Ideal (y = ŷ)")
ax2.set_title(f"Dispersão  (R² = {r2:.3f})")
ax2.set_xlabel("Valor Real")
ax2.set_ylabel("Valor Predito")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 9c. Distribuição dos resíduos
ax3 = fig.add_subplot(gs[1, 1])
residuals = y_test - y_pred
ax3.hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
ax3.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Erro zero")
ax3.set_title(f"Distribuição dos Resíduos  (MAE = {mae:.3f})")
ax3.set_xlabel("Resíduo (real − predito)")
ax3.set_ylabel("Frequência")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.show()
print("\nConcluido.")

