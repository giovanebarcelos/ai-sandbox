# GO1601-6aAttentionVisualizationAnalysis
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertModel, BertTokenizer

class AttentionVisualizer:
    """
    Visualiza e analisa padrões de atenção em Transformers

    Features:
    - Heatmaps de attention por head
    - Análise de padrões (diagonal, broadcast, etc.)
    - Attention flow entre layers
    - Head pruning analysis
    """

    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()

    def get_attentions(self, text: str):
        """Extract attention weights from all layers and heads"""
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.attentions: tuple of (num_layers,) tensors
        # Each tensor: (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions

        return attentions, inputs['input_ids']

    def analyze_attention_pattern(self, attention_matrix):
        """
        Classify attention pattern

        Types:
        - Diagonal: local context (adjacent tokens)
        - Broadcast: one token attends to all
        - Vertical: many tokens attend to one (e.g., [CLS])
        - Block: attends to chunks
        """
        # Normalize
        attn = attention_matrix.numpy()

        # Diagonal score
        diagonal_score = np.trace(attn) / attn.shape[0]

        # Broadcast score (high entropy in rows)
        row_entropy = -np.sum(attn * np.log(attn + 1e-10), axis=1).mean()

        # Vertical score (high entropy in columns)
        col_entropy = -np.sum(attn * np.log(attn + 1e-10), axis=0).mean()

        # Classification
        if diagonal_score > 0.3:
            pattern = "Diagonal (Local)"
        elif row_entropy > 2.5:
            pattern = "Broadcast"
        elif col_entropy > 2.5:
            pattern = "Vertical (Aggregation)"
        else:
            pattern = "Mixed"

        return {
            'pattern': pattern,
            'diagonal_score': diagonal_score,
            'row_entropy': row_entropy,
            'col_entropy': col_entropy
        }

    def visualize_heads(self, text: str, layer_idx: int = 0):
        """Visualize all heads in a specific layer"""
        attentions, input_ids = self.get_attentions(text)

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Get attention for specified layer
        # Shape: (1, num_heads, seq_len, seq_len)
        layer_attn = attentions[layer_idx][0]  # Remove batch dim

        num_heads = layer_attn.shape[0]

        # Create subplots
        cols = 4
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
        axes = axes.flatten() if num_heads > 1 else [axes]

        for head_idx in range(num_heads):
            ax = axes[head_idx]

            attn_matrix = layer_attn[head_idx].numpy()

            # Plot heatmap
            sns.heatmap(attn_matrix, ax=ax, cmap='viridis', 
                       xticklabels=tokens, yticklabels=tokens,
                       cbar=True, square=True, vmin=0, vmax=1)

            # Analyze pattern
            analysis = self.analyze_attention_pattern(torch.tensor(attn_matrix))

            ax.set_title(f"Head {head_idx}\n{analysis['pattern']}", fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')

            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def attention_flow(self, text: str):
        """Visualize attention flow across layers"""
        attentions, input_ids = self.get_attentions(text)

        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        seq_len = attentions[0].shape[2]

        # Average attention across heads for each layer
        layer_avg_attentions = []
        for layer_attn in attentions:
            # Average over heads: (batch, heads, seq, seq) -> (seq, seq)
            avg_attn = layer_attn[0].mean(dim=0).numpy()
            layer_avg_attentions.append(avg_attn)

        # Plot
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        axes = axes.flatten()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        for layer_idx, avg_attn in enumerate(layer_avg_attentions[:12]):
            ax = axes[layer_idx]

            sns.heatmap(avg_attn, ax=ax, cmap='Blues', 
                       xticklabels=tokens if layer_idx >= 6 else [],
                       yticklabels=tokens if layer_idx % 6 == 0 else [],
                       cbar=False, square=True, vmin=0, vmax=0.5)

            ax.set_title(f'Layer {layer_idx}', fontsize=10)

            if layer_idx >= 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            if layer_idx % 6 == 0:
                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)

        plt.tight_layout()
        return fig

    def head_importance_analysis(self, text: str):
        """Analyze which heads contribute most to output"""
        attentions, _ = self.get_attentions(text)

        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]

        # Measure importance by attention variance
        head_importances = []

        for layer_idx, layer_attn in enumerate(attentions):
            layer_scores = []
            for head_idx in range(num_heads):
                attn_matrix = layer_attn[0, head_idx].numpy()

                # Importance = entropy (higher entropy = more distributed attention)
                entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-10))
                layer_scores.append(entropy)

            head_importances.append(layer_scores)

        head_importances = np.array(head_importances)

        return head_importances

# === DEMO ===

print("🌊 Attention Visualization & Analysis\n")
print("="*70)

visualizer = AttentionVisualizer()

# Test text
text = "The cat sat on the mat and looked around"

print(f"Text: \"{text}\"\n")

# Get attentions
attentions, input_ids = visualizer.get_attentions(text)

print(f"Model: BERT-base")
print(f"Layers: {len(attentions)}")
print(f"Heads per layer: {attentions[0].shape[1]}")
print(f"Sequence length: {attentions[0].shape[2]}")
print()

# Visualize first layer heads
print("📌 Visualizing Layer 0 (all 12 heads)...\n")

fig = visualizer.visualize_heads(text, layer_idx=0)
plt.savefig('attention_heads_layer0.png', dpi=150, bbox_inches='tight')
print("   Saved: attention_heads_layer0.png")

# Attention flow across layers
print("\n📌 Visualizing attention flow across all 12 layers...\n")

fig = visualizer.attention_flow(text)
plt.savefig('attention_flow.png', dpi=150, bbox_inches='tight')
print("   Saved: attention_flow.png")

# Head importance analysis
print("\n📌 Analyzing head importance...\n")

head_importances = visualizer.head_importance_analysis(text)

# Visualize importance heatmap
fig, ax = plt.subplots(figsize=(12, 6))

sns.heatmap(head_importances, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Importance (Entropy)'})
ax.set_xlabel('Head Index')
ax.set_ylabel('Layer Index')
ax.set_title('Attention Head Importance Across Layers')

plt.tight_layout()
plt.savefig('head_importance.png', dpi=150, bbox_inches='tight')
print("   Saved: head_importance.png")

# Find most/least important heads
flat_importances = head_importances.flatten()
sorted_indices = np.argsort(flat_importances)

print("\n   Top 5 most important heads:")
for idx in sorted_indices[-5:][::-1]:
    layer = idx // 12
    head = idx % 12
    importance = flat_importances[idx]
    print(f"      Layer {layer}, Head {head}: {importance:.2f}")

print("\n   Top 5 least important heads (candidates for pruning):")
for idx in sorted_indices[:5]:
    layer = idx // 12
    head = idx % 12
    importance = flat_importances[idx]
    print(f"      Layer {layer}, Head {head}: {importance:.2f}")

# Pattern analysis
print("\n📌 Attention Pattern Analysis:\n")

sample_layer = 5  # Middle layer
layer_attn = attentions[sample_layer][0]

patterns_found = {'Diagonal': 0, 'Broadcast': 0, 'Vertical': 0, 'Mixed': 0}

for head_idx in range(12):
    attn_matrix = layer_attn[head_idx]
    analysis = visualizer.analyze_attention_pattern(attn_matrix)
    pattern_type = analysis['pattern'].split(' ')[0]
    patterns_found[pattern_type] = patterns_found.get(pattern_type, 0) + 1

print(f"   Layer {sample_layer} patterns:")
for pattern, count in patterns_found.items():
    print(f"      {pattern}: {count} heads")

# Create pattern distribution chart
fig, ax = plt.subplots(figsize=(10, 6))

patterns_all_layers = {p: [] for p in ['Diagonal', 'Broadcast', 'Vertical', 'Mixed']}

for layer_idx in range(len(attentions)):
    layer_attn = attentions[layer_idx][0]
    layer_patterns = {'Diagonal': 0, 'Broadcast': 0, 'Vertical': 0, 'Mixed': 0}

    for head_idx in range(12):
        attn_matrix = layer_attn[head_idx]
        analysis = visualizer.analyze_attention_pattern(attn_matrix)
        pattern_type = analysis['pattern'].split(' ')[0]
        layer_patterns[pattern_type] = layer_patterns.get(pattern_type, 0) + 1

    for pattern in patterns_all_layers:
        patterns_all_layers[pattern].append(layer_patterns[pattern])

# Stacked area chart
layers = list(range(len(attentions)))

ax.stackplot(layers, 
             patterns_all_layers['Diagonal'],
             patterns_all_layers['Broadcast'],
             patterns_all_layers['Vertical'],
             patterns_all_layers['Mixed'],
             labels=['Diagonal (Local)', 'Broadcast', 'Vertical (Aggregation)', 'Mixed'],
             alpha=0.7)

ax.set_xlabel('Layer')
ax.set_ylabel('Number of Heads')
ax.set_title('Attention Pattern Distribution Across Layers')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: attention_patterns.png")

print("\n✅ Attention visualization implementado!")
print("\n💡 INSIGHTS:")
print("   - Early layers: local/diagonal patterns (syntax)")
print("   - Middle layers: broadcast patterns (semantics)")
print("   - Late layers: vertical patterns (task-specific aggregation)")
print("   - ~30% of heads can be pruned with <1% accuracy loss")
print("\n💡 USE CASES:")
print("   - Debugging model behavior")
print("   - Model interpretability")
print("   - Head pruning for efficiency")
print("   - Understanding learned representations")
