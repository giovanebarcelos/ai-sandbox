# GO2106-VaeCustomFinetuningParaDatasetEspecífico
# ═══════════════════════════════════════════════════════════════════
# VAE CUSTOM - FINE-TUNING PARA DATASET ESPECÍFICO
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ─── 1. PREPARAR DATASET CUSTOM ───
print("📦 Carregando Fashion MNIST (roupas)...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Adicionar dimensão de canal
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"✅ Treino: {x_train.shape}, Teste: {x_test.shape}")

# Classes
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ─── 2. ARQUITETURA VAE MELHORADA ───
latent_dim = 10  # Espaço latente maior para mais complexidade

class Sampling(layers.Layer):
    """Reparameterization trick"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ENCODER
def build_encoder(input_shape=(28, 28, 1), latent_dim=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    return models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

# DECODER
def build_decoder(latent_dim=10):
    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(7 * 7 * 128, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 128))(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    return models.Model(latent_inputs, outputs, name="decoder")

# VAE MODEL
class VAE(models.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # β-VAE: controlar peso do KL loss
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            # Total loss com β
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2)
            )
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        total_loss = reconstruction_loss + self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

# ─── 3. CRIAR E TREINAR VAE ───
print("\n🔨 Construindo VAE...")
encoder = build_encoder(latent_dim=latent_dim)
decoder = build_decoder(latent_dim=latent_dim)
vae = VAE(encoder, decoder, beta=1.0)

# Compilar
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
]

# Treinar
print("\n🚀 Treinando VAE...")
history = vae.fit(
    x_train,
    epochs=50,
    batch_size=128,
    validation_data=(x_test, x_test),
    callbacks=callbacks,
    verbose=1
)

print("✅ Treinamento concluído!")

# ─── 4. VISUALIZAR ESPAÇO LATENTE ───
print("\n📊 Visualizando espaço latente...")

# Encodar dataset de teste
z_mean, _, _ = encoder.predict(x_test, verbose=0)

# PCA para visualizar (se latent_dim > 2)
if latent_dim > 2:
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_mean)
else:
    z_2d = z_mean

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_test, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, ticks=range(10), label='Class')
plt.clim(-0.5, 9.5)
plt.title('Fashion MNIST - Espaço Latente VAE', fontsize=14, fontweight='bold')
plt.xlabel('Dimensão Latente 1')
plt.ylabel('Dimensão Latente 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latent_space_fashion.png', dpi=150)
print("✅ Gráfico salvo: latent_space_fashion.png")

# ─── 5. GERAÇÃO DE NOVAS ROUPAS ───
print("\n👕 Gerando novas roupas...")

n_samples = 15
z_samples = np.random.normal(0, 1, size=(n_samples, latent_dim))
generated_images = decoder.predict(z_samples, verbose=0)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_images[i, :, :, 0], cmap='gray')
    ax.axis('off')
plt.suptitle('Roupas Geradas pelo VAE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('generated_fashion.png', dpi=150)
print("✅ Imagens geradas salvas: generated_fashion.png")

# ─── 6. INTERPOLAÇÃO NO ESPAÇO LATENTE ───
print("\n🔄 Interpolação entre duas peças...")

# Selecionar duas imagens reais
idx1, idx2 = 0, 100  # T-shirt e Ankle boot
img1, img2 = x_test[idx1:idx1+1], x_test[idx2:idx2+1]

# Encodar
z1, _, _ = encoder.predict(img1, verbose=0)
z2, _, _ = encoder.predict(img2, verbose=0)

# Interpolar
steps = 10
interpolations = []
for alpha in np.linspace(0, 1, steps):
    z_interp = (1 - alpha) * z1 + alpha * z2
    img_interp = decoder.predict(z_interp, verbose=0)
    interpolations.append(img_interp[0, :, :, 0])

# Visualizar
fig, axes = plt.subplots(1, steps, figsize=(20, 2))
for i, ax in enumerate(axes):
    ax.imshow(interpolations[i], cmap='gray')
    ax.axis('off')
plt.suptitle(f'Interpolação: {class_names[y_test[idx1]]} → {class_names[y_test[idx2]]}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('interpolation_fashion.png', dpi=150)
print("✅ Interpolação salva: interpolation_fashion.png")

# ─── 7. RECONSTRUÇÃO DE AMOSTRAS ───
print("\n🔧 Comparando originais vs reconstruções...")

n_compare = 10
test_samples = x_test[:n_compare]
reconstructed = vae.predict(test_samples, verbose=0)

fig, axes = plt.subplots(2, n_compare, figsize=(20, 4))
for i in range(n_compare):
    # Original
    axes[0, i].imshow(test_samples[i, :, :, 0], cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontweight='bold')

    # Reconstruído
    axes[1, i].imshow(reconstructed[i, :, :, 0], cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Reconstruído', fontweight='bold')

plt.suptitle('Comparação: Original vs Reconstruído', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reconstruction_comparison.png', dpi=150)
print("✅ Comparação salva: reconstruction_comparison.png")

print("\n" + "="*70)
print("✅ EXERCÍCIO VAE CONCLUÍDO!")
print("="*70)
print("\n📊 Arquivos gerados:")
print("  - latent_space_fashion.png - Visualização do espaço latente")
print("  - generated_fashion.png - 15 peças geradas")
print("  - interpolation_fashion.png - Morfing entre peças")
print("  - reconstruction_comparison.png - Qualidade de reconstrução")

# ─── 8. ANÁLISE DO ESPAÇO LATENTE ───
print("\n📈 Estatísticas do espaço latente:")
print(f"  Média: {z_mean.mean():.4f}")
print(f"  Desvio padrão: {z_mean.std():.4f}")
print(f"  Min: {z_mean.min():.4f}")
print(f"  Max: {z_mean.max():.4f}")
