# GO2101-GanParaMnist
# ═══════════════════════════════════════════════════════════════════
# GAN SIMPLES PARA MNIST
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ───────────────────────────────────────────────────────────────────
# GENERATOR (Gerador)
# ───────────────────────────────────────────────────────────────────

def build_generator(latent_dim=100):
    """
    Entrada: Vetor de ruído (latent_dim dimensões)
    Saída: Imagem 28x28x1 (MNIST)
    """
    model = models.Sequential([
        # Camada densa: ruído → features
        layers.Dense(7*7*128, input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 128)),

        # Upsampling: 7x7 → 14x14
        layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        # Upsampling: 14x14 → 28x28
        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        # Saída: 28x28x1 (imagem)
        layers.Conv2D(1, (5,5), padding='same', activation='tanh')
    ], name='generator')

    return model

# ───────────────────────────────────────────────────────────────────
# DISCRIMINATOR (Discriminador)
# ───────────────────────────────────────────────────────────────────

def build_discriminator():
    """
    Entrada: Imagem 28x28x1
    Saída: Probabilidade de ser real (0-1)
    """
    model = models.Sequential([
        # Convolução: 28x28 → 14x14
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same',
                     input_shape=(28,28,1)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # Convolução: 14x14 → 7x7
        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # Flatten e classificação
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # 0=fake, 1=real
    ], name='discriminator')

    return model

# ───────────────────────────────────────────────────────────────────
# GAN COMPLETO
# ───────────────────────────────────────────────────────────────────

# Criar modelos


if __name__ == "__main__":
    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()

    # Compilar discriminador
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # GAN: gerador + discriminador (discriminador congelado)
    discriminator.trainable = False
    gan_input = layers.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output, name='gan')

    gan.compile(
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        loss='binary_crossentropy'
    )

    print("="*60)
    print("ARQUITETURAS")
    print("="*60)
    generator.summary()
    discriminator.summary()

    # ───────────────────────────────────────────────────────────────────
    # TREINAMENTO (simplificado)
    # ───────────────────────────────────────────────────────────────────

    def train_gan(gan, generator, discriminator, X_train, epochs=100, batch_size=128):
        """
        Treina GAN alternando entre discriminador e gerador
        """
        for epoch in range(epochs):
            # ─── Treinar DISCRIMINADOR ───

            # 1. Selecionar batch de imagens reais
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_images = X_train[idx]

            # 2. Gerar imagens falsas
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_images = generator.predict(noise)

            # 3. Treinar discriminador em reais (label=1) e falsas (label=0)
            d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

            # ─── Treinar GERADOR ───

            # Gerar novo ruído e tentar enganar discriminador (label=1)
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

            # Log
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | D loss: {d_loss:.4f} | G loss: {g_loss:.4f}")

    # Carregar MNIST e treinar
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train.astype('float32') - 127.5) / 127.5  # Normalizar [-1, 1]
    X_train = np.expand_dims(X_train, axis=-1)

    # train_gan(gan, generator, discriminator, X_train, epochs=10000)
