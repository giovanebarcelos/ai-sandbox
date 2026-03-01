# GO1225-35cDataAugmentationPipelineProfissional
# ═══════════════════════════════════════════════════════════════════
# DATA AUGMENTATION PIPELINE COM KERAS
# Técnicas modernas para maximizar variabilidade do dataset
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import matplotlib.pyplot as plt
import cv2

print("🎨 DATA AUGMENTATION PIPELINE AVANÇADO")
print("=" * 70)

# ─── 1. CRIAR IMAGEM DE EXEMPLO ───
print("\n📷 Criando imagem de teste...")

height, width = 224, 224
image = np.zeros((height, width, 3), dtype=np.uint8)

# Adicionar formas geométricas
cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Azul
cv2.circle(image, (180, 80), 30, (0, 255, 0), -1)  # Verde
cv2.ellipse(image, (120, 180), (40, 20), 45, 0, 360, (0, 0, 255), -1)  # Vermelho

print(f"  Imagem criada: {image.shape}")

# ─── 2. PIPELINE LEVE (BASELINE) ───
print("\n🔧 Pipeline LEVE (Baseline):")

datagen_light = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    rotation_range=15
)

print("  ✓ HorizontalFlip")
print("  ✓ Brightness (±20%)")
print("  ✓ Rotation (±15°)")

# ─── 3. PIPELINE MÉDIO (PADRÃO) ───
print("\n🔧 Pipeline MÉDIO (Padrão):")

datagen_medium = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

print("  ✓ HorizontalFlip")
print("  ✓ Rotation (±30°)")
print("  ✓ Shift (±20%)")
print("  ✓ Zoom (±20%)")
print("  ✓ Brightness (±30%)")

# ─── 4. PIPELINE PESADO (AGRESSIVO) ───
print("\n🔧 Pipeline PESADO (Agressivo):")

datagen_heavy = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.5, 1.5],
    fill_mode='reflect'
)

print("  ✓ Horizontal + Vertical Flip")
print("  ✓ Rotation (±45°)")
print("  ✓ Shift (±30%)")
print("  ✓ Zoom (±30%)")
print("  ✓ Shear (±20%)")
print("  ✓ Brightness (±50%)")

# ─── 5. GERAR E VISUALIZAR AUGMENTAÇÕES ───
print("\n📊 Gerando augmentações...")

# Preparar imagem para ImageDataGenerator
img_for_gen = image.reshape((1,) + image.shape)

fig, axes = plt.subplots(4, 6, figsize=(18, 12))

# Original
axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('ORIGINAL', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Linha 1: Pipeline Leve
for i, batch in enumerate(datagen_light.flow(img_for_gen, batch_size=1)):
    if i >= 5:
        break
    aug_img = batch[0].astype('uint8')
    axes[1, i].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'Leve #{i+1}', fontsize=10)
    axes[1, i].axis('off')

# Linha 2: Pipeline Médio
for i, batch in enumerate(datagen_medium.flow(img_for_gen, batch_size=1)):
    if i >= 5:
        break
    aug_img = batch[0].astype('uint8')
    axes[2, i].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    axes[2, i].set_title(f'Médio #{i+1}', fontsize=10)
    axes[2, i].axis('off')

# Linha 3: Pipeline Pesado
for i, batch in enumerate(datagen_heavy.flow(img_for_gen, batch_size=1)):
    if i >= 5:
        break
    aug_img = batch[0].astype('uint8')
    axes[3, i].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    axes[3, i].set_title(f'Pesado #{i+1}', fontsize=10)
    axes[3, i].axis('off')

# Labels das linhas
fig.text(0.02, 0.87, 'Original', fontsize=14, fontweight='bold', rotation=90, va='center')
fig.text(0.02, 0.65, 'LEVE', fontsize=14, fontweight='bold', rotation=90, va='center', color='green')
fig.text(0.02, 0.43, 'MÉDIO', fontsize=14, fontweight='bold', rotation=90, va='center', color='orange')
fig.text(0.02, 0.21, 'PESADO', fontsize=14, fontweight='bold', rotation=90, va='center', color='red')

plt.suptitle('Data Augmentation Pipeline - Comparação de Intensidades', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('augmentation_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Comparação salva: augmentation_comparison.png")

# ─── 6. GUIDELINES DE USO ───
print("\n" + "="*70)
print("📚 GUIDELINES - QUANDO USAR CADA PIPELINE")
print("="*70)

print("\n🟢 PIPELINE LEVE:")
print("  Usar quando:")
print("    • Dataset grande (>50k images)")
print("    • Imagens já bem diversificadas")
print("    • Risco de overfitting baixo")
print("  Ganho típico: +2-5% accuracy")

print("\n🟡 PIPELINE MÉDIO (RECOMENDADO):")
print("  Usar quando:")
print("    • Dataset médio (5k-50k images)")
print("    • Caso de uso geral")
print("    • Balance entre performance e tempo")
print("  Ganho típico: +5-10% accuracy")

print("\n🔴 PIPELINE PESADO:")
print("  Usar quando:")
print("    • Dataset pequeno (<5k images)")
print("    • Alto risco de overfitting")
print("    • Imagens similares/pouca variação")
print("  Ganho típico: +10-20% accuracy")
print("  ⚠️ Cuidado: Pode degradar se excessivo!")

print("\n💡 DICAS PROFISSIONAIS:")
print("  1. Comece com pipeline MÉDIO")
print("  2. Monitore val_loss: se aumentar, reduza augmentation")
print("  3. Augmentation ≠ sempre melhor (pode piorar!)")
print("  4. Valide visualmente: imagens ainda reconhecíveis?")
print("  5. Para medical imaging: augmentation conservador")
print("  6. Para small objects detection: evite crop agressivo")

print("\n✅ PIPELINE DE AUGMENTATION COMPLETO!")
