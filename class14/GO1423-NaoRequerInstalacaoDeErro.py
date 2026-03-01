# GO1423-NãoRequerInstalaçãoDeErro
# ═══════════════════════════════════════════════════════════════════
# ATTENTION MECHANISM - IMPLEMENTAÇÃO DO ZERO
# Mecanismo de atenção para sequencias
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dot, Activation, Concatenate
import matplotlib.pyplot as plt
import seaborn as sns

print("👁️ ATTENTION MECHANISM")
print("=" * 70)

# ─── 1. ATENÇÃO CONCEITO ───
print("\n💡 Conceito de Attention...")
print("  Permite ao modelo 'focar' em partes relevantes da entrada")
print("  Resolve problema de informação em sequências longas")
print("  Bahdanau Attention (2015): attention aditivo")
print("  Luong Attention (2015): attention multiplicativo")

# ─── 2. IMPLEMENTAR ATTENTION LAYER ───
print("\n🏗️ Implementando Attention Layer...")

class AttentionLayer(tf.keras.layers.Layer):
    """
    Bahdanau-style Attention
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query: decoder hidden state (batch, hidden_size)
        # values: encoder outputs (batch, seq_len, hidden_size)

        # Expand query para broadcast
        query_with_time = tf.expand_dims(query, 1)  # (batch, 1, hidden_size)

        # Score
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time) + self.W2(values)
        ))  # (batch, seq_len, 1)

        # Attention weights (softmax)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)

        # Context vector (weighted sum)
        context_vector = attention_weights * values  # (batch, seq_len, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, hidden_size)

        return context_vector, attention_weights

print("✅ AttentionLayer implementada")

# ─── 3. CRIAR DADOS DE EXEMPLO ───
print("\n📦 Criando dados de exemplo...")

# Simulação: Copiar sequência de entrada
# Entrada: [1, 2, 3, 4, 5] → Saída: [1, 2, 3, 4, 5]

np.random.seed(42)

max_len = 10
vocab_size = 20
num_samples = 1000

X_train = []
y_train = []

for _ in range(num_samples):
    seq_len = np.random.randint(5, max_len + 1)
    seq = np.random.randint(1, vocab_size, size=seq_len)

    # Pad
    x_padded = np.pad(seq, (0, max_len - seq_len), constant_values=0)
    y_padded = np.pad(seq, (0, max_len - seq_len), constant_values=0)

    X_train.append(x_padded)
    y_train.append(y_padded)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")

# ─── 4. CONSTRUIR MODELO COM ATTENTION ───
print("\n🏗️ Construindo modelo Seq2Seq com Attention...")

# Encoder
encoder_inputs = Input(shape=(max_len,), name='encoder_input')
encoder_embedding = tf.keras.layers.Embedding(vocab_size, 32)(encoder_inputs)
encoder_lstm = LSTM(64, return_sequences=True, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

print("  Encoder: OK")

# Decoder
decoder_inputs = Input(shape=(max_len,), name='decoder_input')
decoder_embedding = tf.keras.layers.Embedding(vocab_size, 32)(decoder_inputs)
decoder_lstm = LSTM(64, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

print("  Decoder: OK")

# Attention
attention_layer = AttentionLayer(64)

# Aplicar attention em cada timestep do decoder
# Simplificação: usar apenas último estado do decoder
context_vector, attention_weights = attention_layer(state_h, encoder_outputs)

print("  Attention: OK")

# Combinar context com decoder output
decoder_combined = Concatenate()([decoder_outputs, 
                                  tf.keras.layers.RepeatVector(max_len)(context_vector)])

# Output layer
output = Dense(vocab_size, activation='softmax', name='output')(decoder_combined)

model = Model([encoder_inputs, decoder_inputs], output, name='Seq2Seq_Attention')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"\n  Parâmetros: {model.count_params():,}")

# ─── 5. TREINAR ───
print("\n🚀 Treinando modelo...")

# Decoder input: shifted target (teacher forcing)
decoder_input = np.zeros_like(y_train)
decoder_input[:, 1:] = y_train[:, :-1]

# Reshape y para (samples, timesteps, 1)
y_train_reshaped = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

history = model.fit(
    [X_train, decoder_input],
    y_train_reshaped,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 6. VISUALIZAR ATTENTION WEIGHTS ───
print("\n👁️ Visualizando Attention Weights...")

# Criar modelo para extrair attention weights
attention_model = Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=[output, attention_weights]
)

# Testar com uma sequência
test_seq = np.array([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
test_decoder = np.array([[0, 1, 2, 3, 4, 5, 0, 0, 0, 0]])

output_pred, attn_weights = attention_model.predict([test_seq, test_decoder], verbose=0)

print(f"  Attention weights shape: {attn_weights.shape}")

# Plot heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Attention heatmap (simplificado: mostrar apenas primeiros 5x5)
attn_to_plot = attn_weights[0, :5, :].squeeze()

axes[0].imshow(attn_to_plot.T, cmap='YlOrRd', aspect='auto')
axes[0].set_xlabel('Decoder Position', fontsize=11)
axes[0].set_ylabel('Encoder Position', fontsize=11)
axes[0].set_title('Attention Weights Heatmap', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(5))
axes[0].set_yticks(range(5))

# Colorbar
cbar = plt.colorbar(axes[0].images[0], ax=axes[0])
cbar.set_label('Attention', rotation=270, labelpad=15)

# Line plot
for i in range(5):
    axes[1].plot(attn_to_plot[i], marker='o', label=f'Dec {i}')

axes[1].set_xlabel('Encoder Position', fontsize=11)
axes[1].set_ylabel('Attention Weight', fontsize=11)
axes[1].set_title('Attention Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Attention Mechanism Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('attention_weights.png', dpi=150)
print("✅ Attention weights salvos: attention_weights.png")

# ─── 7. COMPARAÇÃO: COM vs SEM ATTENTION ───
print("\n" + "="*70)
print("📊 COMPARAÇÃO: ATTENTION vs BASELINE")
print("="*70)

print("\n🚫 SEM ATTENTION (Vanilla Seq2Seq):")
print("  • Encoder comprime toda entrada em vetor fixo")
print("  • Information bottleneck para sequências longas")
print("  • Performance degrada com comprimento")

print("\n✅ COM ATTENTION:")
print("  • Decoder acessa todas posições do encoder")
print("  • Foco dinâmico em partes relevantes")
print("  • Melhora significativa em sequências longas")
print("  • Interpretability: ver onde modelo 'olha'")

print("\n💡 TIPOS DE ATTENTION:")
print("  • Bahdanau (Additive): score = V * tanh(W1*h + W2*s)")
print("  • Luong (Multiplicative): score = h^T * W * s")
print("  • Dot-Product: score = h^T * s (usado em Transformer)")
print("  • Scaled Dot-Product: score = (h^T * s) / sqrt(d_k)")

print("\n📚 EVOLUÇÃO:")
print("  • 2015: Bahdanau Attention (NMT)")
print("  • 2017: Transformer (Attention is All You Need)")
print("  • 2018: BERT (Bidirectional Transformer)")
print("  • 2020: GPT-3 (175B parâmetros)")
print("  • 2022: ChatGPT (Transformer + RLHF)")

print("\n🎯 APLICAÇÕES:")
print("  • Machine Translation (NMT)")
print("  • Text Summarization")
print("  • Image Captioning (CNN + Attention + LSTM)")
print("  • Speech Recognition")

print("\n✅ ATTENTION MECHANISM COMPLETO!")
