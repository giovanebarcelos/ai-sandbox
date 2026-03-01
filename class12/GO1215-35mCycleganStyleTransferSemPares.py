# GO1215-35mCycleganStyleTransferSemPares
# ══════════════════════════════════════════════════════════════════
# CYCLEGAN - UNPAIRED IMAGE-TO-IMAGE TRANSLATION
# Transformar estilo de imagens sem pares (A→B, B→A)
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print("🎨 CYCLEGAN - UNPAIRED STYLE TRANSFER")
print("=" * 70)

# ─── 1. CRIAR DATASET SINTÉTICO ───
print("\n📦 Gerando domínios A e B...")

def generate_domain_a(num_samples=200):
    """Domínio A: Círculos"""
    images = []
    for _ in range(num_samples):
        img = np.ones((64, 64, 3)) * 0.9

        # Adicionar círculos
        for _ in range(np.random.randint(2, 5)):
            center = (np.random.randint(10, 54), np.random.randint(10, 54))
            radius = np.random.randint(5, 15)
            color = np.random.rand(3) * 0.5

            y, x = np.ogrid[:64, :64]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] = color

        images.append(img)

    return np.array(images, dtype='float32')

def generate_domain_b(num_samples=200):
    """Domínio B: Quadrados"""
    images = []
    for _ in range(num_samples):
        img = np.ones((64, 64, 3)) * 0.9

        # Adicionar quadrados
        for _ in range(np.random.randint(2, 5)):
            x = np.random.randint(5, 50)
            y = np.random.randint(5, 50)
            size = np.random.randint(8, 18)
            color = np.random.rand(3) * 0.5

            img[y:y+size, x:x+size] = color

        images.append(img)

    return np.array(images, dtype='float32')

X_A = generate_domain_a(200)
X_B = generate_domain_b(200)

print(f"  Domain A (Circles): {X_A.shape}")
print(f"  Domain B (Squares): {X_B.shape}")

# ─── 2. CONSTRUIR GENERATORS ───
print("\n🏗️ Construindo Generators...")

def build_generator():
    """U-Net-like generator"""
    inputs = Input(shape=(64, 64, 3))

    # Encoder
    e1 = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    e1 = LeakyReLU(0.2)(e1)

    e2 = Conv2D(128, (4, 4), strides=2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)

    # Decoder
    d1 = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(e2)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)

    d2 = Conv2DTranspose(3, (4, 4), strides=2, padding='same')(d1)
    outputs = Activation('tanh')(d2)

    return Model(inputs, outputs, name='Generator')

# G: A→B, F: B→A
G = build_generator()
F = build_generator()

print(f"  Generator G (A→B): {G.count_params():,} params")
print(f"  Generator F (B→A): {F.count_params():,} params")

# ─── 3. CONSTRUIR DISCRIMINATORS ───
print("\n🏗️ Construindo Discriminators...")

def build_discriminator():
    """PatchGAN discriminator"""
    inputs = Input(shape=(64, 64, 3))

    x = Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, (4, 4), padding='same')(x)

    return Model(inputs, x, name='Discriminator')

D_A = build_discriminator()
D_B = build_discriminator()

print(f"  Discriminator D_A: {D_A.count_params():,} params")
print(f"  Discriminator D_B: {D_B.count_params():,} params")

D_A.compile(optimizer=Adam(0.0002, 0.5), loss='mse')
D_B.compile(optimizer=Adam(0.0002, 0.5), loss='mse')

# ─── 4. CYCLEGAN MODEL ───
print("\n🔄 Construindo CycleGAN...")

# Freeze discriminators para treinar generators
D_A.trainable = False
D_B.trainable = False

# Inputs
input_A = Input(shape=(64, 64, 3))
input_B = Input(shape=(64, 64, 3))

# Forward cycle: A → B → A
fake_B = G(input_A)
reconstructed_A = F(fake_B)

# Backward cycle: B → A → B
fake_A = F(input_B)
reconstructed_B = G(fake_A)

# Identity
identity_A = F(input_A)
identity_B = G(input_B)

# Discriminators
valid_A = D_A(fake_A)
valid_B = D_B(fake_B)

# CycleGAN model
cycle_gan = Model(
    inputs=[input_A, input_B],
    outputs=[valid_A, valid_B, reconstructed_A, reconstructed_B, identity_A, identity_B],
    name='CycleGAN'
)

cycle_gan.compile(
    optimizer=Adam(0.0002, 0.5),
    loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
    loss_weights=[1, 1, 10, 10, 0.5, 0.5]
)

print("  ✓ CycleGAN compiled")

# ─── 5. TREINAR (SIMPLIFICADO) ───
print("\n🚀 Treinando CycleGAN (versão simplificada)...")

epochs = 50
batch_size = 32

# Labels para discriminator
valid = np.ones((batch_size, 8, 8, 1))
fake = np.zeros((batch_size, 8, 8, 1))

for epoch in range(epochs):
    # Selecionar batch aleatório
    idx = np.random.randint(0, X_A.shape[0], batch_size)
    imgs_A = X_A[idx]
    imgs_B = X_B[idx]

    # Gerar imagens fake
    fake_B = G.predict(imgs_A, verbose=0)
    fake_A = F.predict(imgs_B, verbose=0)

    # Treinar discriminators
    d_loss_A = D_A.train_on_batch(imgs_A, valid) + D_A.train_on_batch(fake_A, fake)
    d_loss_B = D_B.train_on_batch(imgs_B, valid) + D_B.train_on_batch(fake_B, fake)

    # Treinar generators
    g_loss = cycle_gan.train_on_batch(
        [imgs_A, imgs_B],
        [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B]
    )

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}/{epochs} - D_loss: {(d_loss_A + d_loss_B)/4:.4f}, G_loss: {g_loss[0]:.4f}")

print("  ✓ Treinamento concluído")

# ─── 6. VISUALIZAR RESULTADOS ───
print("\n🎨 Visualizando transformações...")

fig, axes = plt.subplots(3, 6, figsize=(18, 9))

for i in range(6):
    # A → B → A
    real_A = X_A[i:i+1]
    fake_B = G.predict(real_A, verbose=0)[0]
    reconstructed_A = F.predict(fake_B[np.newaxis, ...], verbose=0)[0]

    axes[0, i].imshow(np.clip(real_A[0], 0, 1))
    axes[0, i].set_title('Real A (Circles)', fontsize=9)
    axes[0, i].axis('off')

    axes[1, i].imshow(np.clip(fake_B, 0, 1))
    axes[1, i].set_title('Fake B (A→B)', fontsize=9, color='green')
    axes[1, i].axis('off')

    axes[2, i].imshow(np.clip(reconstructed_A, 0, 1))
    axes[2, i].set_title('Reconstructed A', fontsize=9)
    axes[2, i].axis('off')

plt.suptitle('CycleGAN: Circles ↔ Squares Translation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('cyclegan_translation.png', dpi=150)
print("✅ Transformações salvas: cyclegan_translation.png")

print("\n💡 CYCLEGAN vs GAN TRADICIONAL:")
print("  • GAN tradicional: Precisa de pares (input, target)")
print("  • CycleGAN: NÃO precisa de pares (unpaired)")
print("  • Cycle Consistency: A→B→A = A")
print("  • Bidirectional: Aprende A→B e B→A simultaneamente")

print("\n🎯 COMPONENTES:")
print("  • 2 Generators: G (A→B), F (B→A)")
print("  • 2 Discriminators: D_A, D_B")
print("  • Cycle Loss: ||F(G(A)) - A|| + ||G(F(B)) - B||")
print("  • Identity Loss: ||F(A) - A|| (preservar cor)")

print("\n📊 APLICAÇÕES REAIS:")
print("  • Horse ↔ Zebra")
print("  • Summer ↔ Winter")
print("  • Photo ↔ Painting (Monet, Van Gogh)")
print("  • Day ↔ Night")
print("  • Sketch ↔ Photo")

print("\n✅ CYCLEGAN COMPLETO!")
