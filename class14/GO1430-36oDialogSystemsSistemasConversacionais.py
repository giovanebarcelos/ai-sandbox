# GO1430-36oDialogSystemsSistemasConversacionais
# ══════════════════════════════════════════════════════════════════
# DIALOG SYSTEMS - CHATBOT COM LSTM
# Sistema conversacional com context tracking
# ══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

print("💬 DIALOG SYSTEMS - CHATBOT")
print("=" * 70)

# ─── 1. DATASET DE DIÁLOGOS ───
print("\n📦 Criando dataset de diálogos...")

np.random.seed(42)

# Pares pergunta-resposta
dialogs = [
    ('ola', 'ola como posso ajudar'),
    ('tudo bem', 'tudo otimo e voce'),
    ('qual seu nome', 'sou um assistente virtual'),
    ('que horas sao', 'consulte seu relogio'),
    ('tchau', 'ate logo tenha um bom dia'),
    ('obrigado', 'de nada estou aqui para ajudar'),
    ('ajuda', 'em que posso ajudar'),
    ('oi', 'ola como vai'),
]

# Expandir
expanded_dialogs = dialogs * 150
np.random.shuffle(expanded_dialogs)

print(f"  Diálogos: {len(expanded_dialogs)}")

# ─── 2. TOKENIZAÇÃO ───
print("\n⚙️ Tokenizando...")

# Vocabulário
vocab = set('<start> <end> <pad>'.split())

for q, a in expanded_dialogs:
    vocab.update(q.split())
    vocab.update(a.split())

vocab = sorted(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)

print(f"  Vocabulário: {vocab_size} palavras")

# Codificar
encoder_input = []
decoder_input = []
decoder_target = []

for q, a in expanded_dialogs:
    q_seq = [word_to_idx[w] for w in q.split()]
    a_seq = [word_to_idx['<start>']] + [word_to_idx[w] for w in a.split()]
    a_out = [word_to_idx[w] for w in a.split()] + [word_to_idx['<end>']]

    encoder_input.append(q_seq)
    decoder_input.append(a_seq)
    decoder_target.append(a_out)

# Pad
max_q_len = max(len(s) for s in encoder_input)
max_a_len = max(len(s) for s in decoder_input)

encoder_input = pad_sequences(encoder_input, maxlen=max_q_len, padding='post', value=word_to_idx['<pad>'])
decoder_input = pad_sequences(decoder_input, maxlen=max_a_len, padding='post', value=word_to_idx['<pad>'])
decoder_target = pad_sequences(decoder_target, maxlen=max_a_len, padding='post', value=word_to_idx['<pad>'])

print(f"  Max question length: {max_q_len}")
print(f"  Max answer length: {max_a_len}")

# ─── 3. CONSTRUIR MODELO ───
print("\n🏗️ Construindo Dialog System...")

latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_q_len,))
encoder_embed = Embedding(vocab_size, 64, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embed)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_a_len,))
decoder_embed = Embedding(vocab_size, 64, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='ChatBot')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"  Parâmetros: {model.count_params():,}")

# ─── 4. TREINAR ───
print("\n🚀 Treinando chatbot...")

history = model.fit(
    [encoder_input, decoder_input],
    decoder_target.reshape(decoder_target.shape[0], decoder_target.shape[1], 1),
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

print(f"  Final accuracy: {history.history['accuracy'][-1]:.4f}")

# ─── 5. INFERENCE ───
print("\n🤖 Criando modo de conversação...")

# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_h = Input(shape=(latent_dim,))
decoder_state_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_h, decoder_state_c]

decoder_embed2 = Embedding(vocab_size, 64, mask_zero=True)
decoder_embedded = decoder_embed2(decoder_inputs)

decoder_outputs2, h, c = LSTM(latent_dim, return_sequences=True, return_state=True)(
    decoder_embedded, initial_state=decoder_states_inputs)
decoder_states2 = [h, c]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

def generate_response(input_text):
    # Tokenizar input
    input_seq = [word_to_idx.get(w, 0) for w in input_text.lower().split()]
    input_seq = pad_sequences([input_seq], maxlen=max_q_len, padding='post', value=word_to_idx['<pad>'])

    # Encoder
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Decoder
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word_to_idx['<start>']

    decoded = []

    for _ in range(max_a_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_idx = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx_to_word.get(sampled_idx, '')

        if sampled_word == '<end>' or sampled_word == '<pad>':
            break

        if sampled_word != '<start>':
            decoded.append(sampled_word)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_idx
        states_value = [h, c]

    return ' '.join(decoded) if decoded else 'nao entendi'

# ─── 6. TESTAR CONVERSAÇÃO ───
print("\n💬 Testando conversação...")

test_inputs = ['ola', 'tudo bem', 'qual seu nome', 'obrigado', 'tchau', 'ajuda']

for user_input in test_inputs:
    response = generate_response(user_input)
    print(f"\n  Usuário: {user_input}")
    print(f"  Bot: {response}")

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

plt.suptitle('Dialog System Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dialog_system_training.png', dpi=150)
print("✅ Training salvo: dialog_system_training.png")

print("\n💡 DIALOG SYSTEMS:")
print("  • Context Tracking: Manter contexto da conversa")
print("  • Intent Recognition: Identificar intenção do usuário")
print("  • Entity Extraction: Extrair informações relevantes")
print("  • Response Generation: Gerar resposta apropriada")

print("\n🎯 ARQUITETURAS:")
print("  • Retrieval-based: Selecionar de respostas pré-definidas")
print("  • Generative: Gerar respostas do zero (Seq2Seq)")
print("  • Hybrid: Combinar ambas abordagens")
print("  • Transformers: GPT, BERT, DialoGPT (SOTA)")

print("\n🔥 APLICAÇÕES:")
print("  • Customer Service: Atendimento automatizado")
print("  • Virtual Assistants: Alexa, Google Assistant")
print("  • Mental Health: Chatbots terapêuticos")
print("  • Education: Tutores virtuais")

print("\n📚 DATASETS:")
print("  • Cornell Movie Dialogs: 220k conversas")
print("  • Ubuntu Dialog Corpus: Technical support")
print("  • PersonaChat: Conversas com personalidade")

print("\n✅ DIALOG SYSTEMS COMPLETO!")
