# GO2112-35aGanDoZero
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("GAN (Generative Adversarial Network) - MNIST")
print("="*70)

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDispositivo: {device}")

# Hiperparâmetros
latent_dim = 100
img_shape = (1, 28, 28)
batch_size = 64
lr = 0.0002
num_epochs = 50

# 1. GENERATOR
class Generator(nn.Module):
    """Gera imagens fake a partir de ruído aleatório"""
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Input: latent_dim (100)
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            # Output: 784 (28x28)
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()  # Output entre -1 e 1
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

# 2. DISCRIMINATOR
class Discriminator(nn.Module):
    """Classifica imagens como real ou fake"""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: 784 (28x28)
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Output: probabilidade (real ou fake)
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 3. INSTANCIAR MODELOS
generator = Generator().to(device)
discriminator = Discriminator().to(device)

print(f"\n📊 Generator: {sum(p.numel() for p in generator.parameters()):,} parâmetros")
print(f"📊 Discriminator: {sum(p.numel() for p in discriminator.parameters()):,} parâmetros")

# 4. LOSS E OPTIMIZERS
adversarial_loss = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 5. CARREGAR DADOS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalizar para [-1, 1]
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\n📚 Dataset: {len(train_dataset)} imagens")
print(f"📦 Batches: {len(train_loader)}")

# 6. TREINAMENTO
print("\n🏋️ Iniciando treinamento...\n")

G_losses = []
D_losses = []

for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(train_loader):
        batch_size_current = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size_current, 1).to(device)
        fake_labels = torch.zeros(batch_size_current, 1).to(device)

        real_imgs = real_imgs.to(device)

        # ==================
        # TREINAR DISCRIMINATOR
        # ==================
        optimizer_D.zero_grad()

        # Loss com imagens reais
        real_validity = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_validity, real_labels)

        # Loss com imagens fake
        z = torch.randn(batch_size_current, latent_dim).to(device)
        fake_imgs = generator(z)
        fake_validity = discriminator(fake_imgs.detach())
        d_fake_loss = adversarial_loss(fake_validity, fake_labels)

        # Total D loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ==================
        # TREINAR GENERATOR
        # ==================
        optimizer_G.zero_grad()

        # Generator quer que discriminator classifique fake como real
        fake_validity = discriminator(fake_imgs)
        g_loss = adversarial_loss(fake_validity, real_labels)

        g_loss.backward()
        optimizer_G.step()

        # Estatísticas
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Salvar losses
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())

    # Gerar amostras a cada 5 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            gen_imgs = generator(z)

            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for idx, ax in enumerate(axes.ravel()):
                img = gen_imgs[idx].cpu().squeeze()
                ax.imshow(img, cmap='gray')
                ax.axis('off')

            plt.suptitle(f'Epoch {epoch+1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'gan_samples_epoch_{epoch+1}.png', dpi=100)
            plt.close()

print("\n✅ Treinamento concluído!")

# 7. PLOTAR LOSSES
plt.figure(figsize=(12, 5))
plt.plot(G_losses, label='Generator Loss', linewidth=2)
plt.plot(D_losses, label='Discriminator Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Losses', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gan_training_losses.png', dpi=150)
print("✅ Gráfico de losses salvo")

# 8. GERAR IMAGENS FINAIS
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)
    gen_imgs = generator(z)

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for idx, ax in enumerate(axes.ravel()):
        img = gen_imgs[idx].cpu().squeeze()
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.suptitle('Imagens Geradas pela GAN', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gan_final_samples.png', dpi=150)
    print("✅ Amostras finais salvas")

print("\n📊 RESULTADO FINAL:")
print(f"  Generator Loss: {G_losses[-1]:.4f}")
print(f"  Discriminator Loss: {D_losses[-1]:.4f}")
print("\n💡 Interpretação:")
print("  - D_loss ≈ 0.5: Discriminator não consegue distinguir fake de real")
print("  - G_loss baixo: Generator consegue enganar o Discriminator")
