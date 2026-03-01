# GO1628-36bAvançadoPositionalEncodingDetalhado
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding:
    """Positional Encoding usando senos e cossenos"""

    @staticmethod
    def get_positional_encoding(seq_len, d_model):
        """
        Calcula positional encoding usando:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros((seq_len, d_model))

        position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # Aplicar sin para posições pares, cos para ímpares
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    @staticmethod
    def visualize_pe(seq_len=100, d_model=128):
        """Visualiza o positional encoding"""
        pe = PositionalEncoding.get_positional_encoding(seq_len, d_model)

        # Plot 1: Heatmap completo
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Heatmap
        sns.heatmap(pe.T, cmap='RdBu_r', center=0, ax=axes[0, 0], 
                    cbar_kws={'label': 'Encoding Value'})
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Embedding Dimension')
        axes[0, 0].set_title('Positional Encoding Heatmap\n(Horizontal: Posição, Vertical: Dimensão)')

        # Primeiras 6 dimensões ao longo das posições
        for dim in range(0, 12, 2):
            axes[0, 1].plot(pe[:, dim], label=f'Dim {dim}', alpha=0.7)
        axes[0, 1].set_xlabel('Position')
        axes[0, 1].set_ylabel('Encoding Value')
        axes[0, 1].set_title('Primeiras Dimensões ao Longo das Posições')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Diferentes frequências (primeiras vs últimas dimensões)
        positions = [0, 10, 25, 50, 75, 99]
        colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

        for pos, color in zip(positions, colors):
            axes[1, 0].plot(pe[pos, :64], label=f'Pos {pos}', 
                           color=color, alpha=0.7)
        axes[1, 0].set_xlabel('Embedding Dimension')
        axes[1, 0].set_ylabel('Encoding Value')
        axes[1, 0].set_title('Encoding em Diferentes Posições (Primeiros 64 dims)')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Similaridade entre posições (produto interno)
        similarity = np.dot(pe, pe.T)
        sns.heatmap(similarity, cmap='coolwarm', center=0, ax=axes[1, 1],
                    cbar_kws={'label': 'Similarity'})
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Position')
        axes[1, 1].set_title('Similaridade Entre Posições\n(Posições próximas = mais similar)')

        plt.tight_layout()
        plt.savefig('positional_encoding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return pe

# Gerar e visualizar


if __name__ == "__main__":
    seq_len = 100
    d_model = 128

    pe = PositionalEncoding.visualize_pe(seq_len, d_model)

    # Análise de propriedades
    print("📊 PROPRIEDADES DO POSITIONAL ENCODING:")
    print(f"\nShape: {pe.shape}")
    print(f"Min value: {pe.min():.4f}")
    print(f"Max value: {pe.max():.4f}")
    print(f"Mean: {pe.mean():.4f}")
    print(f"Std: {pe.std():.4f}")

    # Teste: Posições próximas devem ter encodings similares
    pos_10 = pe[10]
    pos_11 = pe[11]
    pos_50 = pe[50]

    sim_10_11 = np.dot(pos_10, pos_11)
    sim_10_50 = np.dot(pos_10, pos_50)

    print(f"\n🔍 Similaridade entre posições:")
    print(f"Pos 10 vs Pos 11 (adjacentes): {sim_10_11:.4f}")
    print(f"Pos 10 vs Pos 50 (distantes):  {sim_10_50:.4f}")
    print(f"→ Posições próximas são {sim_10_11/sim_10_50:.2f}x mais similares!")

    # Aplicar a embeddings de palavras
    torch.manual_seed(42)
    word_embeddings = torch.randn(seq_len, d_model)
    pe_torch = torch.FloatTensor(pe)

    # Adicionar positional encoding
    embeddings_with_pe = word_embeddings + pe_torch

    print(f"\n✅ Word Embeddings: {word_embeddings.shape}")
    print(f"✅ Positional Encoding: {pe_torch.shape}")
    print(f"✅ Combined: {embeddings_with_pe.shape}")

    # Visualizar efeito em embeddings
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(word_embeddings[:50, :32].numpy(), cmap='viridis', ax=axes[0], cbar=True)
    axes[0].set_title('Word Embeddings\n(Sem informação posicional)')
    axes[0].set_xlabel('Embedding Dim')
    axes[0].set_ylabel('Position')

    sns.heatmap(pe_torch[:50, :32].numpy(), cmap='RdBu_r', center=0, ax=axes[1], cbar=True)
    axes[1].set_title('Positional Encoding\n(Padrão senoidal)')
    axes[1].set_xlabel('Embedding Dim')
    axes[1].set_ylabel('Position')

    sns.heatmap(embeddings_with_pe[:50, :32].numpy(), cmap='viridis', ax=axes[2], cbar=True)
    axes[2].set_title('Combined\n(Word + Position)')
    axes[2].set_xlabel('Embedding Dim')
    axes[2].set_ylabel('Position')

    plt.tight_layout()
    plt.savefig('embedding_with_positional.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✅ Visualizações salvas!")
