# GO2102-715GanCompletoComMonitoram
# ═══════════════════════════════════════════════════════════════════
# DCGAN (DEEP CONVOLUTIONAL GAN) - IMPLEMENTAÇÃO COMPLETA
# Geração de Dígitos MNIST com Monitoramento e Análise
# ═══════════════════════════════════════════════════════════════════

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import os

# ─── 1. CONFIGURAÇÃO ───
print("⚙️ Configurando hiperparâmetros...")

BATCH_SIZE = 256
EPOCHS = 100
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

# Learning rates
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 1e-4

# Criar diretório para salvar
os.makedirs('gan_outputs', exist_ok=True)
os.makedirs('gan_checkpoints', exist_ok=True)

# ─── 2. CARREGAR E PREPARAR DADOS ───
print("\n📦 Carregando MNIST...")
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalizar para [-1, 1] (tanh usa esse range)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

print(f"  Dataset: {train_images.shape}")
print(f"  Range: [{train_images.min():.2f}, {train_images.max():.2f}]")

# Buffer e batch
BUFFER_SIZE = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ─── 3. CONSTRUIR GENERATOR ───
def make_generator_model():
    """
    Generator: Ruído → Imagem 28x28
    Arquitetura DCGAN:
    - Usa Conv2DTranspose (upsampling)
    - BatchNormalization após cada camada (exceto output)
    - LeakyReLU activation
    - Tanh no output (para range [-1, 1])
    """
    model = models.Sequential([
        # Entrada: vetor latente (100,)
        # Projetar e remodelar para 7x7x256
        layers.Dense(7*7*256, use_bias=False, input_shape=(NOISE_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 256)),

        # Upsample 1: 7x7 → 14x14
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        # Upsample 2: 14x14 → 28x28
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        # Output: 28x28x1, tanh para [-1, 1]
        layers.Conv2DTranspose(1, (5, 5), padding='same', use_bias=False, activation='tanh')
    ], name='generator')

    return model

# ─── 4. CONSTRUIR DISCRIMINATOR ───
def make_discriminator_model():
    """
    Discriminator: Imagem → Probabilidade [real/fake]
    Arquitetura DCGAN:
    - Conv2D com strides para downsampling
    - LeakyReLU
    - Dropout para regularização
    - Sem BatchNorm (discriminator)
    """
    model = models.Sequential([
        # Input: 28x28x1

        # Downsample 1: 28x28 → 14x14
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # Downsample 2: 14x14 → 7x7
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # Flatten e classificar
        layers.Flatten(),
        layers.Dense(1)  # Logit (sem sigmoid, usaremos loss from_logits=True)
    ], name='discriminator')

    return model

# ─── 5. INSTANCIAR MODELOS ───
print("\n🔨 Construindo Generator e Discriminator...")
generator = make_generator_model()
discriminator = make_discriminator_model()

print(f"  Generator params: {generator.count_params():,}")
print(f"  Discriminator params: {discriminator.count_params():,}")

# Testar generator
noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)
print(f"  Generator output shape: {generated_image.shape}")

# Testar discriminator
decision = discriminator(generated_image, training=False)
print(f"  Discriminator output shape: {decision.shape}")

# ─── 6. DEFINIR LOSS E OTIMIZADORES ───
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """
    Discriminator quer:
    - Classificar reais como 1
    - Classificar falsas como 0
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Generator quer:
    - Discriminator classifique suas imagens falsas como 1 (enganar)
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Otimizadores
generator_optimizer = tf.keras.optimizers.Adam(GENERATOR_LR, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(DISCRIMINATOR_LR, beta_1=0.5)

# ─── 7. CHECKPOINTS ───
checkpoint_dir = './gan_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# ─── 8. LOOP DE TREINAMENTO ───
# Seed para gerar imagens consistentes durante treino
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

@tf.function
def train_step(images):
    """
    Um step de treinamento GAN
    """
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator cria imagens falsas
        generated_images = generator(noise, training=True)

        # Discriminator avalia reais e falsas
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calcular losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calcular gradientes
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Aplicar gradientes
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    """
    Gera e salva grid de imagens
    """
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.suptitle(f'Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'gan_outputs/image_at_epoch_{epoch:04d}.png', dpi=100)
    plt.close()

# ─── 9. FUNÇÃO DE TREINAMENTO PRINCIPAL ───
def train(dataset, epochs):
    """
    Treina GAN por N épocas
    """
    print("\n🚀 Iniciando treinamento GAN...\n")

    # Histórico de losses
    history = {
        'gen_loss': [],
        'disc_loss': [],
        'epoch_times': []
    }

    for epoch in range(epochs):
        start = time.time()

        epoch_gen_loss = []
        epoch_disc_loss = []

        # Treinar em todos os batches
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss.append(gen_loss.numpy())
            epoch_disc_loss.append(disc_loss.numpy())

        # Médias da época
        avg_gen_loss = np.mean(epoch_gen_loss)
        avg_disc_loss = np.mean(epoch_disc_loss)
        history['gen_loss'].append(avg_gen_loss)
        history['disc_loss'].append(avg_disc_loss)

        # Tempo
        epoch_time = time.time() - start
        history['epoch_times'].append(epoch_time)

        # Gerar imagens a cada 5 épocas
        if (epoch + 1) % 5 == 0:
            display.clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed)

        # Salvar checkpoint a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Log
        print(f'Época {epoch+1:03d}/{epochs} | '
              f'Gen Loss: {avg_gen_loss:.4f} | '
              f'Disc Loss: {avg_disc_loss:.4f} | '
              f'Tempo: {epoch_time:.2f}s')

        # Detectar mode collapse
        if avg_gen_loss < 0.5 and avg_disc_loss > 2.0:
            print("  ⚠️ Possível mode collapse detectado!")

    # Gerar imagens finais
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

    return history

# ─── 10. TREINAR! ───
history = train(train_dataset, EPOCHS)

print("\n✅ Treinamento concluído!")

# ─── 11. VISUALIZAR HISTÓRICO DE TREINAMENTO ───
print("\n📊 Gerando gráficos de análise...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss ao longo das épocas
epochs_range = range(1, EPOCHS + 1)
axes[0].plot(epochs_range, history['gen_loss'], label='Generator Loss', color='blue', alpha=0.7)
axes[0].plot(epochs_range, history['disc_loss'], label='Discriminator Loss', color='red', alpha=0.7)
axes[0].set_title('Losses durante Treinamento', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Tempo por época
axes[1].plot(epochs_range, history['epoch_times'], color='green', alpha=0.7)
axes[1].set_title('Tempo por Época', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Tempo (s)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gan_outputs/training_history.png', dpi=150)
print("  ✓ Histórico salvo: gan_outputs/training_history.png")

# ─── 12. ANÁLISE DE QUALIDADE - INTERPOLAÇÃO ───
print("\n🔄 Gerando interpolações no espaço latente...")

def interpolate_points(p1, p2, n_steps=10):
    """Interpola entre dois pontos no espaço latente"""
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = [(1.0 - ratio) * p1 + ratio * p2 for ratio in ratios]
    return np.array(vectors)

# Dois pontos aleatórios
point1 = tf.random.normal([1, NOISE_DIM])
point2 = tf.random.normal([1, NOISE_DIM])

# Interpolar
interpolated = interpolate_points(point1[0], point2[0], n_steps=10)
interpolated = tf.convert_to_tensor(interpolated, dtype=tf.float32)

# Gerar imagens
interpolated_images = generator(interpolated, training=False)

# Visualizar
fig, axes = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    axes[i].imshow(interpolated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    axes[i].axis('off')
plt.suptitle('Interpolação Suave no Espaço Latente', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gan_outputs/latent_interpolation.png', dpi=150)
print("  ✓ Interpolação salva: gan_outputs/latent_interpolation.png")

# ─── 13. ANÁLISE DE QUALIDADE - VARIAÇÕES ───
print("\n🎲 Gerando variações com diferentes seeds...")

# Gerar múltiplas versões do "mesmo" dígito
n_variations = 20
variations_noise = tf.random.normal([n_variations, NOISE_DIM])
variations_images = generator(variations_noise, training=False)

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(variations_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    ax.axis('off')
plt.suptitle('20 Dígitos Gerados Aleatoriamente', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gan_outputs/random_variations.png', dpi=150)
print("  ✓ Variações salvas: gan_outputs/random_variations.png")

# ─── 14. ANÁLISE DE MODE COLLAPSE ───
print("\n🔬 Analisando diversidade (Mode Collapse)...")

# Gerar 100 imagens
n_samples = 100
test_noise = tf.random.normal([n_samples, NOISE_DIM])
test_images = generator(test_noise, training=False)

# Calcular variância pixel-wise
test_images_array = test_images.numpy().reshape(n_samples, -1)
pixel_variance = np.var(test_images_array, axis=0).mean()

print(f"  Variância média dos pixels: {pixel_variance:.6f}")
if pixel_variance < 0.01:
    print("  ⚠️ Baixa diversidade - possível mode collapse")
elif pixel_variance > 0.1:
    print("  ✅ Boa diversidade - GAN saudável")
else:
    print("  ⚡ Diversidade moderada")

# ─── 15. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ TREINAMENTO E ANÁLISE GAN CONCLUÍDOS!")
print("="*70)

print("\n📊 Estatísticas Finais:")
print(f"  - Épocas treinadas: {EPOCHS}")
print(f"  - Loss final do Generator: {history['gen_loss'][-1]:.4f}")
print(f"  - Loss final do Discriminator: {history['disc_loss'][-1]:.4f}")
print(f"  - Tempo médio por época: {np.mean(history['epoch_times']):.2f}s")
print(f"  - Tempo total: {sum(history['epoch_times'])/60:.1f} min")
print(f"  - Variância pixel-wise: {pixel_variance:.6f}")

print("\n📁 Arquivos gerados:")
print("  - gan_outputs/image_at_epoch_XXXX.png - Progresso do treinamento")
print("  - gan_outputs/training_history.png - Gráficos de loss")
print("  - gan_outputs/latent_interpolation.png - Interpolação suave")
print("  - gan_outputs/random_variations.png - 20 dígitos gerados")
print("  - gan_checkpoints/ - Checkpoints do modelo")

print("\n💡 Dicas para melhorar:")
if history['disc_loss'][-1] < 0.5:
    print("  → Discriminator muito forte: reduzir learning rate ou adicionar dropout")
if history['gen_loss'][-1] > 3.0:
    print("  → Generator fraco: aumentar capacidade ou reduzir dropout")
if pixel_variance < 0.05:
    print("  → Mode collapse: tentar diferentes learning rates, adicionar noise, ou usar WGAN")

# ─── 16. FUNÇÃO PARA GERAR NOVAS IMAGENS ───
def generate_new_digits(n=10):
    """Função auxiliar para gerar novos dígitos"""
    noise = tf.random.normal([n, NOISE_DIM])
    generated = generator(noise, training=False)

    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].imshow(generated[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        axes[i].axis('off')

    plt.suptitle(f'{n} Novos Dígitos Gerados', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("\n🎨 Para gerar novos dígitos, use: generate_new_digits(10)")
