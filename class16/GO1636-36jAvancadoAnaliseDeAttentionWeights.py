# GO1636-36jAvançadoAnáliseDeAttentionWeights
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    """Visualizar attention weights do BERT"""

    def __init__(self, model_name='bert-base-uncased'):
        print(f"🔄 Carregando {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        print("✅ Modelo carregado!")

    def get_attentions(self, text):
        """Obter attention weights para texto"""

        inputs = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # Tuple of (batch, num_heads, seq_len, seq_len)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return attentions, tokens

    def visualize_attention_head(self, text, layer=11, head=0):
        """Visualizar attention de uma cabeça específica"""

        attentions, tokens = self.get_attentions(text)

        # Selecionar layer e head
        attention = attentions[layer][0, head].numpy()  # (seq_len, seq_len)

        # Plot
        plt.figure(figsize=(12, 10))

        sns.heatmap(attention, 
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Attention Weight'},
                   square=True,
                   linewidths=0.5)

        plt.title(f'Attention Weights - Layer {layer}, Head {head}\n' +
                 f'Text: "{text}"',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Key (Attended To)', fontsize=12)
        plt.ylabel('Query (Attending From)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(f'attention_layer{layer}_head{head}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Análise
        self._analyze_attention_pattern(attention, tokens)

    def _analyze_attention_pattern(self, attention, tokens):
        """Analisar padrões de atenção"""

        print("\n📊 ANÁLISE DE ATENÇÃO:\n")

        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            # Top-3 tokens que este token atende
            top_3_idx = attention[i].argsort()[-3:][::-1]

            print(f"'{token}' presta mais atenção em:")
            for idx in top_3_idx:
                if tokens[idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                    print(f"   - '{tokens[idx]}': {attention[i, idx]:.4f}")

    def compare_all_heads(self, text, layer=11):
        """Comparar todas as cabeças de um layer"""

        attentions, tokens = self.get_attentions(text)

        num_heads = attentions[0].shape[1]

        # Calcular grid
        cols = 4
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        axes = axes.flatten()

        for head in range(num_heads):
            attention = attentions[layer][0, head].numpy()

            sns.heatmap(attention,
                       xticklabels=tokens if head < 4 else False,  # Labels apenas primeiras
                       yticklabels=tokens if head % 4 == 0 else False,
                       cmap='viridis',
                       cbar=True,
                       ax=axes[head],
                       square=True)

            axes[head].set_title(f'Head {head}', fontsize=11)
            if head < 4:
                axes[head].tick_params(axis='x', labelsize=8, rotation=45)
            if head % 4 == 0:
                axes[head].tick_params(axis='y', labelsize=8)

        # Esconder axes extras
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'All Attention Heads - Layer {layer}\nText: "{text}"',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'all_heads_layer{layer}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def compare_layers(self, text, head=0):
        """Comparar attention através das layers"""

        attentions, tokens = self.get_attentions(text)

        num_layers = len(attentions)

        # Selecionar algumas layers representativas
        selected_layers = [0, 3, 6, 9, 11]

        fig, axes = plt.subplots(1, len(selected_layers), figsize=(20, 5))

        for idx, layer in enumerate(selected_layers):
            attention = attentions[layer][0, head].numpy()

            sns.heatmap(attention,
                       xticklabels=tokens if idx == 0 else False,
                       yticklabels=tokens if idx == 0 else False,
                       cmap='plasma',
                       cbar=True,
                       ax=axes[idx],
                       square=True,
                       vmin=0,
                       vmax=0.5)  # Normalizar escala

            axes[idx].set_title(f'Layer {layer}', fontsize=12)
            if idx == 0:
                axes[idx].tick_params(axis='both', labelsize=8, rotation=45)

        plt.suptitle(f'Attention Evolution Across Layers - Head {head}\n' +
                    f'Text: "{text}"',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'layers_comparison_head{head}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def attention_flow(self, text):
        """Visualizar fluxo de atenção agregado"""

        attentions, tokens = self.get_attentions(text)

        # Agregar attention: média sobre layers e heads
        all_attentions = torch.stack([att[0] for att in attentions])  # (layers, heads, seq, seq)
        avg_attention = all_attentions.mean(dim=[0, 1]).numpy()  # (seq, seq)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Heatmap completo
        sns.heatmap(avg_attention,
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='RdYlGn_r',
                   cbar_kws={'label': 'Avg Attention'},
                   ax=axes[0],
                   square=True,
                   linewidths=0.5)
        axes[0].set_title('Average Attention (All Layers & Heads)', fontsize=14)
        axes[0].set_xlabel('Key')
        axes[0].set_ylabel('Query')
        axes[0].tick_params(axis='both', labelsize=9, rotation=45)

        # Grafo de fluxo (top connections)
        threshold = 0.1  # Mostrar apenas conexões fortes

        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if i != j and avg_attention[i, j] > threshold:
                    # Desenhar seta
                    axes[1].annotate('',
                                    xy=(j, len(tokens)-i-1),
                                    xytext=(i, len(tokens)-i-1),
                                    arrowprops=dict(
                                        arrowstyle='->',
                                        lw=avg_attention[i, j]*10,
                                        alpha=avg_attention[i, j],
                                        color='red'
                                    ))

        # Plot tokens
        for i, token in enumerate(tokens):
            axes[1].text(i, len(tokens)-i-1, token, 
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                        fontsize=10)

        axes[1].set_xlim(-1, len(tokens))
        axes[1].set_ylim(-1, len(tokens))
        axes[1].set_aspect('equal')
        axes[1].axis('off')
        axes[1].set_title(f'Attention Flow (threshold={threshold})', fontsize=14)

        plt.suptitle(f'Aggregated Attention Analysis\nText: "{text}"',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('attention_flow.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Estatísticas
        print("\n📊 ESTATÍSTICAS DE ATENÇÃO:")
        print(f"   Atenção média: {avg_attention.mean():.4f}")
        print(f"   Atenção máxima: {avg_attention.max():.4f}")
        print(f"   Atenção mínima: {avg_attention.min():.4f}")

        # Tokens mais atendidos
        total_attention = avg_attention.sum(axis=0)
        top_attended_idx = total_attention.argsort()[-5:][::-1]

        print(f"\n🎯 Top-5 tokens mais atendidos:")
        for idx in top_attended_idx:
            if tokens[idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                print(f"   - '{tokens[idx]}': {total_attention[idx]:.4f}")

# Inicializar visualizador


if __name__ == "__main__":
    visualizer = AttentionVisualizer()

    # Teste 1: Attention de uma cabeça específica
    print("\n" + "="*80)
    print("TESTE 1: Visualizar Attention Head Específica")
    print("="*80)

    text1 = "The cat sat on the mat because it was tired."
    visualizer.visualize_attention_head(text1, layer=11, head=5)

    # Teste 2: Comparar todas as heads
    print("\n" + "="*80)
    print("TESTE 2: Comparar Todas as Heads")
    print("="*80)

    text2 = "Machine learning is transforming the world."
    visualizer.compare_all_heads(text2, layer=11)

    # Teste 3: Comparar layers
    print("\n" + "="*80)
    print("TESTE 3: Evolução Através das Layers")
    print("="*80)

    text3 = "Python is a powerful programming language."
    visualizer.compare_layers(text3, head=0)

    # Teste 4: Fluxo de atenção agregado
    print("\n" + "="*80)
    print("TESTE 4: Fluxo de Atenção Agregado")
    print("="*80)

    text4 = "The quick brown fox jumps over the lazy dog."
    visualizer.attention_flow(text4)

    print("\n✅ Análise completa!")
    print("\n📊 INSIGHTS:")
    print("   - Primeiras layers: atenção sintática (adjacência)")
    print("   - Layers intermediárias: relações semânticas")
    print("   - Últimas layers: contexto global e tarefa")
    print("   - Diferentes heads capturam diferentes aspectos")
    print("   - [CLS] acumula informação global")
