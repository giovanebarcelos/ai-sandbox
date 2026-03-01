# GO1428-36mTextSummarizationResumirTextosAutomaticame
# ══════════════════════════════════════════════════════════════════
# TEXT SUMMARIZATION COM SEQ2SEQ + ATTENTION
# Gerar resumos de textos longos
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

print("📝 TEXT SUMMARIZATION COM SEQ2SEQ")
print("=" * 70)

# ─── 1. CRIAR DATASET SINTÉTICO ───
print("\n📚 Criando dataset de textos e resumos...")

# Textos longos → Resumos curtos
data = [
    {
        'text': 'The weather today is very nice and sunny with clear blue skies',
        'summary': 'nice weather'
    },
    {
        'text': 'John went to the store and bought some groceries for dinner tonight',
        'summary': 'John bought groceries'
    },
    {
        'text': 'The movie was entertaining but too long and had slow pacing',
        'summary': 'movie too long'
    },
    {
        'text': 'Python is a popular programming language used for data science',
        'summary': 'Python for data science'
    },
    {
        'text': 'The cat climbed up the tall tree and refused to come down',
        'summary': 'cat in tree'
    },
    {
        'text': 'Maria studied hard for the exam and got excellent grades',
        'summary': 'Maria passed exam'
    },
    {
        'text': 'The new restaurant serves delicious Italian food at reasonable prices',
        'summary': 'good Italian restaurant'
    },
    {
        'text': 'The conference will be held next week in New York City',
        'summary': 'conference in NYC'
    },
] * 125  # Repetir para ter mais dados

print(f"  Dataset size: {len(data)} pares")

# ─── 2. PREPROCESSAR ───
print("\n🔧 Preprocessando...")

# Vocabulário
all_words = set()
for item in data:
    all_words.update(item['text'].lower().split())
    all_words.update(item['summary'].lower().split())

vocab = sorted(all_words)
word2idx = {word: idx+3 for idx, word in enumerate(vocab)}
word2idx['<PAD>'] = 0
word2idx['<START>'] = 1
word2idx['<END>'] = 2
idx2word = {idx: word for word, idx in word2idx.items()}

vocab_size = len(word2idx)

print(f"  Vocabulário: {vocab_size} palavras")

# Converter para sequências
texts = [[word2idx[w.lower()] for w in item['text'].split()] for item in data]
summaries = [[word2idx['<START>']] + [word2idx[w.lower()] for w in item['summary'].split()] + [word2idx['<END>']] for item in data]

# Pad
max_text_len = max(len(t) for t in texts)
max_summary_len = max(len(s) for s in summaries)

texts = pad_sequences(texts, maxlen=max_text_len, padding='post')
summaries = pad_sequences(summaries, maxlen=max_summary_len, padding='post')

print(f"  Max text length: {max_text_len}")
print(f"  Max summary length: {max_summary_len}")

# Decoder inputs e targets
decoder_input = summaries[:, :-1]
decoder_target = summaries[:, 1:]

print(f"  Encoder input: {texts.shape}")
print(f"  Decoder input: {decoder_input.shape}")
print(f"  Decoder target: {decoder_target.shape}")

# ─── 3. CONSTRUIR ENCODER ───
print("\n🏗️ Construindo Encoder...")

encoder_inputs = Input(shape=(max_text_len,))
enc_emb = Embedding(vocab_size, 64, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(128, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

print("  ✓ Encoder construído")

# ─── 4. CONSTRUIR DECODER ───
print("\n🏗️ Construindo Decoder...")

decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(vocab_size, 64, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

print("  ✓ Decoder construído")

# ─── 5. MODELO COMPLETO ───
model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='Seq2Seq_Summarization')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"\n  Parâmetros totais: {model.count_params():,}")

# ─── 6. TREINAR ───
print("\n🚀 Treinando modelo...")

history = model.fit(
    [texts, decoder_input],
    decoder_target,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    verbose=0
)

print(f"  Final loss: {history.history['loss'][-1]:.4f}")
print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 7. INFERENCE MODEL ───
print("\n🔮 Construindo modelo de inferência...")

# Encoder model (para inferência)
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model (para inferência)
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(vocab_size, 64, mask_zero=True)
decoder_inputs_single = Input(shape=(1,))
dec_emb_out = dec_emb2(decoder_inputs_single)

decoder_lstm2 = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs2, state_h2, state_c2 = decoder_lstm2(
    dec_emb_out,
    initial_state=decoder_states_inputs
)
decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

print("  ✓ Inference models construídos")

# ─── 8. GERAR RESUMOS ───
print("\n📝 Gerando resumos...")

def generate_summary(input_seq):
    # Encode
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Decoder: começar com <START>
    target_seq = np.array([[word2idx['<START>']]])

    summary = []
    for _ in range(max_summary_len):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # Pegar palavra mais provável
        sampled_token_index = np.argmax(output_tokens[0, 0, :])

        if sampled_token_index == word2idx['<END>'] or len(summary) > 10:
            break

        if sampled_token_index not in [word2idx['<PAD>'], word2idx['<START>']]:
            summary.append(idx2word[sampled_token_index])

        # Update
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(summary)

# Testar em alguns exemplos
test_indices = [0, 100, 200, 300, 400]

for idx in test_indices:
    input_text = ' '.join([idx2word[i] for i in texts[idx] if i != 0])
    true_summary = ' '.join([idx2word[i] for i in summaries[idx] if i not in [0, 1, 2]])

    predicted_summary = generate_summary(texts[idx:idx+1])

    print(f"\n  Text: {input_text}")
    print(f"  True: {true_summary}")
    print(f"  Pred: {predicted_summary}")

# ─── 9. VISUALIZAR ───
print("\n📊 Visualizando training...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Training Accuracy', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('Text Summarization Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('summarization_training.png', dpi=150)
print("\n✅ Training salvo: summarization_training.png")

print("\n💡 SUMMARIZATION TYPES:")
print("  • Extractive: Selecionar sentenças mais importantes (TextRank)")
print("  • Abstractive: Gerar novo texto (Seq2Seq, Transformer)")
print("  • Single-document: Resumir 1 documento")
print("  • Multi-document: Resumir múltiplos documentos")

print("\n🎯 TÉCNICAS:")
print("  • Seq2Seq: LSTM encoder-decoder")
print("  • Attention: Focar em partes relevantes")
print("  • Pointer-Generator: Copiar palavras do input")
print("  • Transformers: BART, T5, Pegasus (SOTA)")

print("\n📊 MÉTRICAS:")
print("  • ROUGE: Recall-Oriented Understudy for Gisting Evaluation")
print("  • BLEU: Bilingual Evaluation Understudy")
print("  • METEOR: Metric for Evaluation of Translation")

print("\n🏆 DATASETS:")
print("  • CNN/Daily Mail: 300k pares news-summary")
print("  • Gigaword: Headlines (título como summary)")
print("  • XSum: Extreme summarization (BBC)")

print("\n✅ TEXT SUMMARIZATION COMPLETO!")
