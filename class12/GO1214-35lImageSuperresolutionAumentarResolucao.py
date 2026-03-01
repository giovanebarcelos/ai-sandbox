# GO1214-35lImageSuperresolutionAumentarResolução
# ══════════════════════════════════════════════════════════════════
# IMAGE SUPER-RESOLUTION COM CNN
# Aumentar resolução de imagens mantendo qualidade
# ══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, Add
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

print("🔍 IMAGE SUPER-RESOLUTION")
print("=" * 70)

# ─── 1. CARREGAR DADOS ───
print("\n📦 Carregando CIFAR-10...")

(X_train, _), (X_test, _) = cifar10.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Usar apenas subset
X_train = X_train[:1000]
X_test = X_test[:100]

print(f"  Train: {X_train.shape}")
print(f"  Test: {X_test.shape}")

# ─── 2. CRIAR LOW-RES IMAGES ───
print("\n📉 Gerando imagens de baixa resolução...")

def downscale_image(img, scale=2):
    """
    Reduz resolução da imagem
    """
    h, w = img.shape[:2]
    new_h, new_w = h // scale, w // scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def upscale_image(img, scale=2):
    """
    Aumenta resolução da imagem (bicubic baseline)
    """
    h, w = img.shape[:2]
    new_h, new_w = h * scale, w * scale
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# Criar pares LR-HR
scale_factor = 2
X_train_lr = np.array([downscale_image(img, scale_factor) for img in X_train])
X_test_lr = np.array([downscale_image(img, scale_factor) for img in X_test])

print(f"  HR shape: {X_train.shape}")
print(f"  LR shape: {X_train_lr.shape}")

# ─── 3. CONSTRUIR MODELO SRCNN ───
print("\n🏗️ Construindo SRCNN (Super-Resolution CNN)...")

# Input é LR image upscaled
input_lr = Input(shape=(32, 32, 3))

# SRCNN architecture
x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_lr)
x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
output = Conv2D(3, (5, 5), activation='linear', padding='same')(x)

model_srcnn = Model(inputs=input_lr, outputs=output, name='SRCNN')

model_srcnn.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(f"  Parâmetros: {model_srcnn.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando SRCNN...")

# Upscale LR images para tamanho original (input para SRCNN)
X_train_lr_upscaled = np.array([upscale_image(img, scale_factor) for img in X_train_lr])
X_test_lr_upscaled = np.array([upscale_image(img, scale_factor) for img in X_test_lr])

history = model_srcnn.fit(
    X_train_lr_upscaled,
    X_train,  # Target é HR original
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    verbose=0
)

print(f"  Final MSE: {history.history['loss'][-1]:.6f}")
print(f"  Final MAE: {history.history['mae'][-1]:.6f}")

# ─── 5. COMPARAR MÉTODOS ───
print("\n📊 Comparando métodos de super-resolution...")

fig, axes = plt.subplots(3, 5, figsize=(18, 11))

psnr_bicubic_list = []
psnr_srcnn_list = []
ssim_bicubic_list = []
ssim_srcnn_list = []

for i in range(5):
    idx = i

    # Original HR
    hr_original = X_test[idx]

    # LR
    lr = X_test_lr[idx]

    # Bicubic upscaling
    bicubic = upscale_image(lr, scale_factor)

    # SRCNN
    lr_upscaled = upscale_image(lr, scale_factor)
    srcnn = model_srcnn.predict(lr_upscaled[np.newaxis, ...], verbose=0)[0]
    srcnn = np.clip(srcnn, 0, 1)

    # Calcular métricas
    psnr_bicubic = psnr(hr_original, bicubic, data_range=1.0)
    psnr_srcnn = psnr(hr_original, srcnn, data_range=1.0)

    ssim_bicubic = ssim(hr_original, bicubic, data_range=1.0, channel_axis=2)
    ssim_srcnn = ssim(hr_original, srcnn, data_range=1.0, channel_axis=2)

    psnr_bicubic_list.append(psnr_bicubic)
    psnr_srcnn_list.append(psnr_srcnn)
    ssim_bicubic_list.append(ssim_bicubic)
    ssim_srcnn_list.append(ssim_srcnn)

    # Visualizar
    axes[0, i].imshow(lr)
    axes[0, i].set_title(f'LR ({lr.shape[0]}x{lr.shape[1]})', fontsize=9)
    axes[0, i].axis('off')

    axes[1, i].imshow(bicubic)
    axes[1, i].set_title(f'Bicubic\nPSNR: {psnr_bicubic:.2f}', fontsize=9)
    axes[1, i].axis('off')

    axes[2, i].imshow(srcnn)
    axes[2, i].set_title(f'SRCNN\nPSNR: {psnr_srcnn:.2f}', fontsize=9, color='green', fontweight='bold')
    axes[2, i].axis('off')

plt.suptitle('Image Super-Resolution Comparison (2x)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('super_resolution_comparison.png', dpi=150)
print("✅ Comparação salva: super_resolution_comparison.png")

# ─── 6. MÉTRICAS ───
print("\n" + "="*70)
print("📊 MÉTRICAS MÉDIAS")
print("="*70)

print(f"\n📈 PSNR (Peak Signal-to-Noise Ratio):")
print(f"  Bicubic: {np.mean(psnr_bicubic_list):.2f} dB")
print(f"  SRCNN:   {np.mean(psnr_srcnn_list):.2f} dB")
print(f"  Melhoria: +{np.mean(psnr_srcnn_list) - np.mean(psnr_bicubic_list):.2f} dB")

print(f"\n📊 SSIM (Structural Similarity):")
print(f"  Bicubic: {np.mean(ssim_bicubic_list):.4f}")
print(f"  SRCNN:   {np.mean(ssim_srcnn_list):.4f}")
print(f"  Melhoria: +{np.mean(ssim_srcnn_list) - np.mean(ssim_bicubic_list):.4f}")

# Gráfico de métricas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['Bicubic', 'SRCNN'], 
           [np.mean(psnr_bicubic_list), np.mean(psnr_srcnn_list)],
           color=['steelblue', 'green'], alpha=0.7)
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('PSNR Comparison (Higher is better)', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(['Bicubic', 'SRCNN'],
           [np.mean(ssim_bicubic_list), np.mean(ssim_srcnn_list)],
           color=['steelblue', 'green'], alpha=0.7)
axes[1].set_ylabel('SSIM')
axes[1].set_title('SSIM Comparison (Higher is better)', fontsize=12, fontweight='bold')
axes[1].set_ylim(0.8, 1.0)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Super-Resolution Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('super_resolution_metrics.png', dpi=150)
print("✅ Métricas salvas: super_resolution_metrics.png")

print("\n💡 MÉTODOS DE SUPER-RESOLUTION:")
print("  • Bicubic/Bilinear: Interpolação clássica (baseline)")
print("  • SRCNN (2014): Primeira CNN para SR")
print("  • VDSR (2016): Very Deep SR (20 layers)")
print("  • ESPCN (2016): Sub-pixel convolution")
print("  • SRGAN (2017): GAN-based, fotorealismo")
print("  • EDSR (2017): Enhanced Deep SR (SOTA)")

print("\n🎯 APLICAÇÕES:")
print("  • Médica: Melhorar scans de baixa resolução")
print("  • Satélite: Aumentar resolução de imagens espaciais")
print("  • Streaming: Upscale de vídeos")
print("  • Fotografia: Zoom digital inteligente")

print("\n📊 MÉTRICAS:")
print("  • PSNR: Relação sinal/ruído (dB)")
print("  • SSIM: Similaridade estrutural (0-1)")
print("  • LPIPS: Perceptual similarity (deep features)")

print("\n✅ SUPER-RESOLUTION COMPLETO!")
