# GO1420-36eOpcionalEncoderdecoderLstmParaTradução
# ═══════════════════════════════════════════════════════════════════
# ENCODER-DECODER LSTM PARA TRADUÇÃO (SEQUENCE-TO-SEQUENCE)
# Traduzir sequências numéricas (conceito de NMT simplificado)
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt

print("🌍 ENCODER-DECODER LSTM PARA TRADUÇÃO")
print("=" * 70)

# ─── 1. CRIAR DATASET SINTÉTICO ───
print("\n📦 Criando dataset de tradução sintético...")
print("   Tarefa: Inverter sequências numéricas")
print("   Exemplo: [1, 2, 3] → [3, 2, 1]")

np.random.seed(42)

# Parâmetros
num_samples = 5000
max_len = 10
num_tokens = 20  # Vocabulário: 0-19

# Gerar sequências de entrada
encoder_input_data = []
decoder_input_data = []
decoder_target_data = []

for _ in range(num_samples):
    # Sequência de entrada (aleatória)
    seq_len = np.random.randint(3, max_len + 1)
    input_seq = np.random.randint(1, num_tokens, size=seq_len)

    # Sequência de saída (invertida)
    target_seq = input_seq[::-1]

    # Decoder input: <START> + target[:-1]
    # Decoder target: target + <END>
    decoder_input = np.concatenate([[0], target_seq[:-1]])  # 0 = <START>
    decoder_target = np.concatenate([target_seq, [0]])  # 0 = <END> (reuso)

    # Pad sequences
    input_padded = np.pad(input_seq, (0, max_len - seq_len), constant_values=0)
    decoder_input_padded = np.pad(decoder_input, (0, max_len - len(decoder_input)), constant_values=0)
    decoder_target_padded = np.pad(decoder_target, (0, max_len - len(decoder_target)), constant_values=0)

    encoder_input_data.append(input_padded)
    decoder_input_data.append(decoder_input_padded)
    decoder_target_data.append(decoder_target_padded)

encoder_input_data = np.array(encoder_input_data)
decoder_input_data = np.array(decoder_input_data)
decoder_target_data = np.array(decoder_target_data)

print(f"  Samples: {num_samples}")
print(f"  Encoder input shape: {encoder_input_data.shape}")
print(f"  Decoder input shape: {decoder_input_data.shape}")
print(f"  Decoder target shape: {decoder_target_data.shape}")

# One-hot encode
encoder_input_onehot = tf.keras.utils.to_categorical(encoder_input_data, num_tokens)
decoder_input_onehot = tf.keras.utils.to_categorical(decoder_input_data, num_tokens)
decoder_target_onehot = tf.keras.utils.to_categorical(decoder_target_data, num_tokens)

# ─── 2. CONSTRUIR ENCODER-DECODER ───
print("\n🏗️ Construindo modelo Encoder-Decoder...")

# ENCODER
encoder_inputs = Input(shape=(max_len, num_tokens), name='encoder_input')
encoder_lstm = LSTM(128, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# DECODER
decoder_inputs = Input(shape=(max_len, num_tokens), name='decoder_input')
decoder_lstm = LSTM(128, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_tokens, activation='softmax', name='decoder_output')
decoder_outputs = decoder_dense(decoder_outputs)

# MODEL
model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='Seq2Seq')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 3. TREINAR ───
print("\n🚀 Treinando modelo...")

history = model.fit(
    [encoder_input_onehot, decoder_input_onehot],
    decoder_target_onehot,
    batch_size=64,
    epochs=20,
    validation_split=0.2,
    verbose=0
)

print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final val_accuracy: {history.history['val_accuracy'][-1]:.4f}")

# ─── 4. MODELO DE INFERÊNCIA ───
print("\n🔮 Criando modelo de inferência...")

# Encoder (extrai estados)
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder (recebe estados anteriores)
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf
)

print("  ✓ Encoder model criado")
print("  ✓ Decoder model criado")

# ─── 5. FUNÇÃO DE DECODIFICAÇÃO ───
def decode_sequence(input_seq):
    """
    Decodifica uma sequência usando o modelo treinado
    """
    # Encode input
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Target sequence inicial (<START>)
    target_seq = np.zeros((1, max_len, num_tokens))
    target_seq[0, 0, 0] = 1  # <START> token

    decoded_seq = []

    for i in range(max_len):
        # Predict next token
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample token
        sampled_token_index = np.argmax(output_tokens[0, i, :])

        if sampled_token_index == 0 and i > 0:  # <END> token
            break

        decoded_seq.append(sampled_token_index)

        # Update target sequence
        target_seq[0, i+1, sampled_token_index] = 1

        # Update states
        states_value = [h, c]

    return decoded_seq

# ─── 6. TESTAR PREDIÇÕES ───
print("\n👁️ Testando predições...")

# Selecionar 10 amostras
num_tests = 10
test_indices = np.random.choice(len(encoder_input_onehot), num_tests, replace=False)

correct = 0

for idx in test_indices:
    input_seq = encoder_input_onehot[idx:idx+1]
    decoded = decode_sequence(input_seq)

    # Original
    original = encoder_input_data[idx][encoder_input_data[idx] > 0]
    expected = original[::-1]

    # Check
    is_correct = np.array_equal(decoded, expected)
    correct += is_correct

    symbol = "✓" if is_correct else "✗"
    print(f"  {symbol} Input: {list(original)} → Pred: {decoded} | Expected: {list(expected)}")

accuracy = correct / num_tests
print(f"\n  Accuracy: {accuracy:.1%} ({correct}/{num_tests})")

# ─── 7. VISUALIZAR TRAINING ───
print("\n📊 Visualizando training...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Encoder-Decoder Training', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('seq2seq_training.png', dpi=150)
print("✅ Training salvo: seq2seq_training.png")

print("\n💡 ENCODER-DECODER ARCHITECTURE:")
print("  1. ENCODER: Processa input → states (h, c)")
print("  2. DECODER: Recebe states → gera output sequencial")
print("  3. Teacher Forcing: Durante treino, usa ground truth como input")
print("  4. Inferência: Decodifica token por token (autoregressive)")

print("\n📚 APLICAÇÕES REAIS:")
print("  • Neural Machine Translation (NMT)")
print("  • Chatbots e QA systems")
print("  • Text Summarization")
print("  • Image Captioning (CNN encoder + LSTM decoder)")

print("\n✅ ENCODER-DECODER COMPLETO!")
