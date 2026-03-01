# GO1218-35pAutoaugmentDataAugmentationAutomático
# ══════════════════════════════════════════════════════════════════
# AUTOAUGMENT - BUSCA AUTOMÁTICA DE AUGMENTATION
# Encontrar melhores políticas de augmentation
# ══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

print("🔍 AUTOAUGMENT - AUTOMATIC DATA AUGMENTATION")
print("=" * 70)

# ─── 1. CARREGAR DADOS ───
print("\n📦 Carregando CIFAR-10...")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Subset
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. DEFINIR AUGMENTATION OPERATIONS ───
print("\n🎨 Definindo operações de augmentation...")

def rotate(img, magnitude):
    angle = magnitude * 30  # -30 a +30 graus
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def shear_x(img, magnitude):
    shear = magnitude * 0.3
    h, w = img.shape[:2]
    M = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def translate_x(img, magnitude):
    tx = int(magnitude * img.shape[1] * 0.3)
    M = np.array([[1, 0, tx], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)

def adjust_brightness(img, magnitude):
    factor = 1.0 + magnitude * 0.9
    return np.clip(img * factor, 0, 1)

def adjust_contrast(img, magnitude):
    factor = 1.0 + magnitude * 0.9
    mean = img.mean()
    return np.clip((img - mean) * factor + mean, 0, 1)

def adjust_saturation(img, magnitude):
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + magnitude), 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb.astype(np.float32) / 255.0

augmentation_ops = [
    ('rotate', rotate),
    ('shearX', shear_x),
    ('translateX', translate_x),
    ('brightness', adjust_brightness),
    ('contrast', adjust_contrast),
    ('saturation', adjust_saturation)
]

print(f"  Operações disponíveis: {len(augmentation_ops)}")

# ─── 3. DEFINIR POLÍTICAS ───
print("\n📜 Definindo políticas de augmentation...")

# Políticas (simplificadas, normalmente encontradas por RL)
policies = [
    # Política 1: Geometria
    [(augmentation_ops[0], 0.8), (augmentation_ops[1], 0.6)],

    # Política 2: Cor
    [(augmentation_ops[3], 0.9), (augmentation_ops[4], 0.7)],

    # Política 3: Misto
    [(augmentation_ops[2], 0.5), (augmentation_ops[5], 0.8)],
]

print(f"  Número de políticas: {len(policies)}")

# ─── 4. APLICAR AUGMENTATION ───
print("\n⚙️ Aplicando augmentation com políticas...")

def apply_policy(img, policy):
    for (op_name, op_func), magnitude in policy:
        img = op_func(img, magnitude)
    return img

X_train_aug = []
for img in X_train[:1000]:
    # Escolher política aleatória
    policy = policies[np.random.randint(0, len(policies))]
    aug_img = apply_policy(img.copy(), policy)
    X_train_aug.append(aug_img)

X_train_aug = np.array(X_train_aug)

print(f"  Augmented data: {X_train_aug.shape}")

# ─── 5. TREINAR SEM VS COM AUTOAUGMENT ───
print("\n🚀 Comparando modelos...")

# Modelo sem augmentation
model_no_aug = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
], name='NoAugment')

model_no_aug.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_no_aug.fit(X_train[:1000], y_train[:1000], epochs=20, batch_size=64, verbose=0)

acc_no_aug = model_no_aug.evaluate(X_test, y_test, verbose=0)[1]

print(f"  Sem augmentation: {acc_no_aug:.4f}")

# Modelo com AutoAugment
model_aug = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
], name='AutoAugment')

model_aug.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar com dados originais + augmentados
X_combined = np.concatenate([X_train[:1000], X_train_aug])
y_combined = np.concatenate([y_train[:1000], y_train[:1000]])

model_aug.fit(X_combined, y_combined, epochs=20, batch_size=64, verbose=0)

acc_aug = model_aug.evaluate(X_test, y_test, verbose=0)[1]

print(f"  Com AutoAugment: {acc_aug:.4f}")
print(f"  Melhoria: +{(acc_aug - acc_no_aug)*100:.1f}%")

# ─── 6. VISUALIZAR ───
print("\n📊 Visualizando augmentações...")

fig, axes = plt.subplots(len(policies) + 1, 5, figsize=(15, 12))

# Original
for i in range(5):
    axes[0, i].imshow(X_train[i])
    axes[0, i].set_title('Original', fontsize=10)
    axes[0, i].axis('off')

# Cada política
for policy_idx, policy in enumerate(policies):
    for i in range(5):
        aug_img = apply_policy(X_train[i].copy(), policy)
        axes[policy_idx + 1, i].imshow(np.clip(aug_img, 0, 1))
        axes[policy_idx + 1, i].set_title(f'Policy {policy_idx + 1}', fontsize=10)
        axes[policy_idx + 1, i].axis('off')

plt.suptitle('AutoAugment Policies', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('autoaugment_policies.png', dpi=150)
print("✅ Políticas salvas: autoaugment_policies.png")

# Comparação
fig, ax = plt.subplots(figsize=(8, 6))

models = ['No Augmentation', 'AutoAugment']
accs = [acc_no_aug, acc_aug]

ax.bar(models, accs, color=['steelblue', 'green'], alpha=0.7)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('AutoAugment Performance', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)

for i, acc in enumerate(accs):
    ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('autoaugment_comparison.png', dpi=150)
print("✅ Comparação salva: autoaugment_comparison.png")

print("\n💡 AUTOAUGMENT:")
print("  • Busca automática: RL para encontrar melhores políticas")
print("  • Políticas: Sequências de operações + magnitudes")
print("  • Dataset-specific: Otimizado para cada dataset")
print("  • SOTA: Melhora generalização sem alterar arquitetura")

print("\n🎯 VARIAÇÕES:")
print("  • RandAugment: Random magnitude, mais simples")
print("  • AugMax: Adversarial augmentation")
print("  • TrivialAugment: Sem busca, política fixa")
print("  • AutoAugment-RL: Reinforcement Learning")

print("\n📊 BENEFÍCIOS:")
print("  • +2-4% accuracy em CIFAR/ImageNet")
print("  • Melhor generalização")
print("  • Reduz overfitting")
print("  • Transfer learning: Políticas transferem entre datasets")

print("\n✅ AUTOAUGMENT COMPLETO!")
