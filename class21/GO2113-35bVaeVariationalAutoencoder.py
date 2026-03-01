# GO2113-35bVaeVariationalAutoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("="*70)
print("VAE (Variational Autoencoder) - MNIST")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparâmetros
latent_dim = 20
batch_size = 128
learning_rate = 1e-3
num_epochs = 30

# 1. MODELO VAE
class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Latent space (mean e log_variance)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()  # Output [0, 1]
        )

    def encode(self, x):
        """Mapeia x para distribuição no espaço latente"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Reconstrói x a partir de z"""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# 2. LOSS FUNCTION
def vae_loss(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL Divergence: KL(q(z|x) || p(z))
    # onde p(z) = N(0, I)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

# 3. TREINAR
model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\n📊 Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
print(f"📚 Dataset: {len(train_dataset)} imagens")
print("\n🏋️ Treinando VAE...\n")

train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_bce = 0
    epoch_kld = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Calcular loss
        loss, bce, kld = vae_loss(recon_batch, data, mu, logvar)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_bce += bce.item()
        epoch_kld += kld.item()

    avg_loss = epoch_loss / len(train_dataset)
    avg_bce = epoch_bce / len(train_dataset)
    avg_kld = epoch_kld / len(train_dataset)

    train_losses.append(avg_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f})')

    # Visualizar reconstruções
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            # Pegar 8 imagens de teste
            data_sample = next(iter(train_loader))[0][:8].to(device)
            recon, _, _ = model(data_sample)

            fig, axes = plt.subplots(2, 8, figsize=(16, 4))

            for i in range(8):
                # Original
                axes[0, i].imshow(data_sample[i].cpu().squeeze(), cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original', fontsize=12)

                # Reconstruída
                axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel('Reconstruída', fontsize=12)

            plt.suptitle(f'VAE Reconstruction - Epoch {epoch+1}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'vae_recon_epoch_{epoch+1}.png', dpi=100)
            plt.close()

print("\n✅ Treinamento concluído!")

# 4. GERAR NOVAS IMAGENS
model.eval()
with torch.no_grad():
    # Amostrar do espaço latente
    z = torch.randn(64, latent_dim).to(device)
    samples = model.decode(z)

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for idx, ax in enumerate(axes.ravel()):
        img = samples[idx].cpu().view(28, 28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.suptitle('Imagens Geradas pela VAE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('vae_generated_samples.png', dpi=150)
    print("✅ Amostras geradas salvas")

# 5. VISUALIZAR ESPAÇO LATENTE (2D)
if latent_dim == 2:
    model.eval()
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    z_list = []
    labels_list = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            z_list.append(mu.cpu())
            labels_list.append(labels)

    z = torch.cat(z_list).numpy()
    labels = torch.cat(labels_list).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=5)
    plt.colorbar(scatter, label='Dígito')
    plt.xlabel('Dimensão Latente 1')
    plt.ylabel('Dimensão Latente 2')
    plt.title('Espaço Latente da VAE', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vae_latent_space.png', dpi=150)
    print("✅ Espaço latente visualizado")

print("\n📊 VAE COMPLETO!")
print("\n💡 VAE vs GAN:")
print("  VAE: Probabilístico, estável, boa reconstrução")
print("  GAN: Adversarial, instável, imagens mais nítidas")
