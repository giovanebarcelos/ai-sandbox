# GO1429-36nNeuralMachineTranslationTraduçãoAutomática
# ══════════════════════════════════════════════════════════════════
# NEURAL MACHINE TRANSLATION (NMT)
# Seq2Seq com Attention para tradução
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

print("🌍 NEURAL MACHINE TRANSLATION (NMT)")
print("=" * 70)

# ─── 1. DATASET SINTÉTICO ───
print("\n📦 Gerando dataset de tradução...")

np.random.seed(42)

# Pares português → inglês (simplificado)
pairs = [
    ('ola mundo', 'hello world'),
    ('bom dia', 'good morning'),
    ('boa noite', 'good night'),
    ('como vai', 'how are you'),
    ('tudo bem', 'all good'),
    ('ate logo', 'see you'),
    ('obrigado', 'thank you'),
    ('de nada', 'you welcome'),
]

# Expandir dataset
expanded_pairs = pairs * 100
np.random.shuffle(expanded_pairs)

print(f"  Pares de tradução: {len(expanded_pairs)}")

# ─── 2. TOKENIZAÇÃO ───
print("\n⚙️ Tokenizando...")

# Vocabulários
src_vocab = set()
tgt_vocab = set('<start> <end>'.split())

for src, tgt in expanded_pairs:
    src_vocab.update(src.split())
    tgt_vocab.update(tgt.split())

src_vocab = ['<pad>'] + sorted(src_vocab)
tgt_vocab = ['<pad>'] + sorted(tgt_vocab)

src_word_to_idx = {w: i for i, w in enumerate(src_vocab)}
tgt_word_to_idx = {w: i for i, w in enumerate(tgt_vocab)}
tgt_idx_to_word = {i: w for w, i in tgt_word_to_idx.items()}

print(f"  Source vocab: {len(src_vocab)}")
print(f"  Target vocab: {len(tgt_vocab)}")

# Codificar
encoder_input = []
decoder_input = []
decoder_target = []

for src, tgt in expanded_pairs:
    src_seq = [src_word_to_idx[w] for w in src.split()]
    tgt_seq = [tgt_word_to_idx['<start>']] + [tgt_word_to_idx[w] for w in tgt.split()]
    tgt_out = [tgt_word_to_idx[w] for w in tgt.split()] + [tgt_word_to_idx['<end>']]

    encoder_input.append(src_seq)
    decoder_input.append(tgt_seq)
    decoder_target.append(tgt_out)

# Pad
max_src_len = max(len(s) for s in encoder_input)
max_tgt_len = max(len(s) for s in decoder_input)

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tgt_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tgt_len, padding='post')

print(f"  Max source length: {max_src_len}")
print(f"  Max target length: {max_tgt_len}")

# ─── 3. CONSTRUIR MODELO SEQ2SEQ ───
print("\n🏗️ Construindo Seq2Seq...")

latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_src_len,))
encoder_embed = Embedding(len(src_vocab), 64)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embed)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_tgt_len,))
decoder_embed = Embedding(len(tgt_vocab), 64)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
decoder_dense = Dense(len(tgt_vocab), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Modelo completo
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando NMT...")

history = model.fit(
    [encoder_input, decoder_input],
    decoder_target.reshape(decoder_target.shape[0], decoder_target.shape[1], 1),
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"  Final loss: {history.history['loss'][-1]:.4f}")
print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 5. INFERENCE MODEL ───
print("\n🔮 Criando modelo de inferência...")

# Encoder
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embed2 = Embedding(len(tgt_vocab), 64)
decoder_embedded = decoder_embed2(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = LSTM(latent_dim, return_sequences=True, return_state=True)(
    decoder_embedded, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# ─── 6. TRADUZIR ───
print("\n🌐 Traduzindo...")

def translate(input_seq):
    # Encoder
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Decoder: começar com <start>
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tgt_word_to_idx['<start>']

    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tgt_idx_to_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence) > max_tgt_len:
            stop_condition = True
        else:
            decoded_sentence.append(sampled_word)

        # Update
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# Testar
test_samples = [0, 1, 2, 3, 4, 5]

for idx in test_samples:
    input_seq = encoder_input[idx:idx+1]
    translated = translate(input_seq)

    # Decodificar input
    src_text = ' '.join([src_vocab[i] for i in encoder_input[idx] if i > 0])

    # Ground truth
    tgt_text = ' '.join([tgt_vocab[i] for i in decoder_target[idx] if i > 0 and tgt_vocab[i] != '<end>'])

    print(f"\n  Português: {src_text}")
    print(f"  Tradução: {translated}")
    print(f"  Ground Truth: {tgt_text}")

# ─── 7. VISUALIZAR ───
print("\n📊 Visualizando training...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Neural Machine Translation Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('nmt_training.png', dpi=150)
print("✅ Training salvo: nmt_training.png")

print("\n💡 NMT ARCHITECTURE:")
print("  • Encoder: Processa sequência fonte")
print("  • Decoder: Gera sequência alvo")
print("  • Context Vector: Resume fonte")
print("  • Attention: Melhor alinhamento (não implementado aqui)")

print("\n🎯 TÉCNICAS:")
print("  • Beam Search: Múltiplas hipóteses")
print("  • Length Normalization: Penalizar traduções curtas")
print("  • Coverage: Evitar repetições")
print("  • Transformers: SOTA atual")

print("\n📚 DATASETS:")
print("  • WMT: Workshop on Machine Translation")
print("  • Europarl: Parallel corpus (7 languages)")
print("  • OpenSubtitles: Movie subtitles")

print("\n✅ NMT COMPLETO!")
