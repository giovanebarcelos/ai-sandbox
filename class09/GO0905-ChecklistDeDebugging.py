# GO0905-ChecklistDeDebugging
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: RECONHECIMENTO DE DÍGITOS (MNIST)
# Etapa 1: CARREGAR E PREPARAR DADOS
# ═══════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
print("="*70)
print("CLASSIFICAÇÃO DE DÍGITOS MANUSCRITOS - MNIST")
print("="*70)
# CARREGAR E EXPLORAR DATASET
print("\n1. Carregando dataset MNIST...")
# MNIST: 70,000 imagens 28×28 pixels de dígitos 0-9
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy().astype('float32') / 255.0  # Normalizar para [0, 1]
y = mnist.target.to_numpy().astype(int)
print(f"✅ Dataset carregado!")
print(f"   Shape: {X.shape}")
print(f"   Labels: {np.unique(y)}")
# VISUALIZAR EXEMPLOS
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()
for i in range(10):
    # Pegar primeira imagem de cada dígito
    idx = np.where(y == i)[0][0]
    img = X[idx].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Dígito {i}')
    axes[i].axis('off')
plt.suptitle('Exemplos de Dígitos MNIST', fontsize=14)
plt.tight_layout()
plt.show()
# PREPARAR DADOS
print("\n2. Preparando dados...")
# Usar subset menor para treino mais rápido (10,000 amostras)
X_subset = X[:10000]
y_subset = y[:10000]
# Split train/val/test: 60% / 20% / 20%
X_train, X_temp, y_train, y_temp = train_test_split(
    X_subset, y_subset, test_size=0.4, random_state=42, stratify=y_subset
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print(f"   Train: {X_train.shape[0]} amostras")
print(f"   Val:   {X_val.shape[0]} amostras")
print(f"   Test:  {X_test.shape[0]} amostras")
# Normalizar (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
# One-hot encode labels
def one_hot_encode(y, num_classes=10):
    """Converte labels para one-hot encoding"""
    n = len(y)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot
y_train_oh = one_hot_encode(y_train)
y_val_oh = one_hot_encode(y_val)
y_test_oh = one_hot_encode(y_test)
print(f"\n✅ Dados preparados!")
print(f"   Input: 784 features (28×28 pixels)")
print(f"   Output: 10 classes (dígitos 0-9)")

# ───────────────────────────────────────────────────────────────────
# ✅ CHECKPOINT ETAPA 1:
# ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("✅ CHECKPOINT ETAPA 1 - VALIDAÇÕES")
print("="*70)
assert X_train.shape == (6000, 784), f"Shape errado! Esperado (6000, 784), obtido {X_train.shape}"
assert y_train_oh.shape == (6000, 10), f"Shape errado! Esperado (6000, 10), obtido {y_train_oh.shape}"
assert X_train.min() >= -3 and X_train.max() <= 3, "Normalização falhou! Valores fora do range esperado"
print("✓ X_train shape: OK")
print("✓ y_train_oh shape: OK")
print("✓ Normalização: OK")
print("✓ Split treino/val/test: OK")
print("\n🎉 Etapa 1 completa! Prossiga para Slide 22 (Etapa 2)")
