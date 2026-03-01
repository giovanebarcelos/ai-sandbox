# GO1627-36aAvançadoSelfattention
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SelfAttention(nn.Module):
    """Implementação simplificada de Self-Attention"""
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projeções lineares para Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, embed_dim = x.shape

        # 1. Projetar para Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, embed_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Reshape para multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Calcular attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        # 4. Aplicar attention aos valores
        attention_output = torch.matmul(attention_weights, V)

        # 5. Concatenar heads e projetar
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embed_dim)
        output = self.W_o(attention_output)

        if return_attention:
            return output, attention_weights
        return output

# Teste com sentença de exemplo
torch.manual_seed(42)
embed_dim = 64
seq_len = 8
batch_size = 1

# Criar embedding simulado
words = ["O", "gato", "comeu", "o", "rato", "que", "estava", "dormindo"]
x = torch.randn(batch_size, seq_len, embed_dim)

# Criar modelo
attention = SelfAttention(embed_dim, num_heads=4)

# Forward pass
output, attention_weights = attention(x, return_attention=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Visualizar attention weights para primeira cabeça
plt.figure(figsize=(10, 8))
att_head_0 = attention_weights[0, 0].detach().numpy()

sns.heatmap(att_head_0, 
            xticklabels=words,
            yticklabels=words,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Attention Weight'})

plt.title('Self-Attention Weights - Head 0\n"Para cada palavra (linha), onde ela presta atenção (coluna)"')
plt.xlabel('Keys (Palavras de Origem)')
plt.ylabel('Queries (Palavras que Atendem)')
plt.tight_layout()
plt.savefig('self_attention_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Análise das atenções
print("\n📊 Análise de Atenção:")
for i, word in enumerate(words):
    top_3_idx = att_head_0[i].argsort()[-3:][::-1]
    top_3_words = [words[idx] for idx in top_3_idx]
    top_3_scores = [att_head_0[i, idx] for idx in top_3_idx]
    print(f"\n'{word}' presta mais atenção em:")
    for w, s in zip(top_3_words, top_3_scores):
        print(f"  - '{w}': {s:.3f}")

# Comparar diferentes números de heads
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, num_heads in enumerate([1, 4, 8]):
    att_model = SelfAttention(embed_dim, num_heads=num_heads)
    _, att_weights = att_model(x, return_attention=True)

    # Média sobre todas as heads
    att_mean = att_weights[0].mean(dim=0).detach().numpy()

    sns.heatmap(att_mean,
                xticklabels=words,
                yticklabels=words,
                cmap='viridis',
                annot=True,
                fmt='.2f',
                ax=axes[i],
                cbar=True)
    axes[i].set_title(f'{num_heads} Head(s)')
    axes[i].set_xlabel('Keys')
    axes[i].set_ylabel('Queries')

plt.suptitle('Comparação: Efeito do Número de Attention Heads', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('multi_head_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualizações salvas!")
print("   - self_attention_visualization.png")
print("   - multi_head_comparison.png")
