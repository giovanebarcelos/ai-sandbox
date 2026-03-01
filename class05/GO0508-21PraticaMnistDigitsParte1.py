# GO0508-21PráticaMnistDigitsParte1
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS (MNIST)
# ═══════════════════════════════════════════════════════════════════

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

print("="*60)
print("MNIST - CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS")
print("="*60)
print("\nCarregando dataset...")

# ───────────────────────────────────────────────────────────────────
# CARREGAR MNIST
# ───────────────────────────────────────────────────────────────────

# MNIST: 70.000 imagens 28x28 de dígitos manuscritos (0-9)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

print(f"\nDataset carregado!")
print(f"Shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Total de exemplos: {len(X)}")

# ───────────────────────────────────────────────────────────────────
# VISUALIZAR ALGUMAS IMAGENS
# ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f'Label: {y.iloc[i]}')
    ax.axis('off')
plt.suptitle('Exemplos do MNIST Dataset')
plt.tight_layout()
plt.show()

# ───────────────────────────────────────────────────────────────────
# ANÁLISE EXPLORATÓRIA
# ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ANÁLISE EXPLORATÓRIA")
print("="*60)

# Distribuição de classes
unique, counts = np.unique(y, return_counts=True)
print("\nDistribuição de classes:")
for digit, count in zip(unique, counts):
    print(f"  Dígito {digit}: {count} exemplos ({count/len(y)*100:.1f}%)")

# Valores dos pixels
print(f"\nValores dos pixels:")
print(f"  Min: {X.min().min()}")
print(f"  Max: {X.max().max()}")
print(f"  Média: {X.mean().mean():.2f}")

# Plotar distribuição
plt.figure(figsize=(10, 5))
plt.bar(unique, counts, color='steelblue', alpha=0.8)
plt.xlabel('Dígito')
plt.ylabel('Número de Exemplos')
plt.title('Distribuição de Classes no MNIST')
plt.xticks(unique)
plt.grid(axis='y', alpha=0.3)
plt.show()

print("\n✅ Dataset carregado e explorado!")
