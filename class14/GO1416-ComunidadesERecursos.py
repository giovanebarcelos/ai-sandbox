# GO1416-ComunidadesERecursos
# ═══════════════════════════════════════════════════════════════════
# GERAÇÃO DE TEXTO COM LSTM - CHARACTER-LEVEL LANGUAGE MODEL
# Treinar modelo para gerar texto estilo Shakespeare
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.callbacks import LambdaCallback
import random

print("📝 GERAÇÃO DE TEXTO COM LSTM")
print("=" * 70)

# ─── 1. PREPARAR TEXTO ───
print("\n📚 Preparando corpus de texto...")

# Texto de exemplo (Shakespeare-like)
text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life."""

text = text.lower()
print(f"  Texto length: {len(text)} caracteres")

# Criar mapeamento char -> índice
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
print(f"  Vocabulário: {vocab_size} caracteres únicos")
print(f"  Chars: {chars[:20]}...")

# ─── 2. CRIAR SEQUÊNCIAS DE TREINAMENTO ───
print("\n🔨 Criando sequências...")

seq_length = 40  # Usar 40 caracteres para prever o próximo
step = 3  # Stride

sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    seq = text[i:i + seq_length]
    next_char = text[i + seq_length]
    sequences.append(seq)
    next_chars.append(next_char)

print(f"  Sequências criadas: {len(sequences)}")
print(f"  Exemplo:")
print(f"    Input: '{sequences[0]}'")
print(f"    Target: '{next_chars[0]}'")

# ─── 3. CODIFICAR SEQUÊNCIAS ───
print("\n🔢 Codificando sequências...")

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool_)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ─── 4. CONSTRUIR MODELO ───
print("\n🏗️ Construindo modelo LSTM...")

model = Sequential([
    LSTM(128, input_shape=(seq_length, vocab_size)),
    Dense(vocab_size, activation='softmax')
], name='CharLSTM')

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 5. FUNÇÃO DE SAMPLING ───
def sample_with_temperature(preds, temperature=1.0):
    """
    Sample próximo caractere com temperatura
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ─── 6. CALLBACK PARA GERAR TEXTO ───
def on_epoch_end(epoch, _):
    print(f"\n─── Geração após epoch {epoch + 1} ───")

    # Seed aleatório
    start_index = random.randint(0, len(text) - seq_length - 1)
    seed = text[start_index:start_index + seq_length]

    for temperature in [0.2, 0.5, 1.0]:
        print(f"\nTemperature {temperature}:")
        print(f"Seed: '{seed}'")

        generated = seed
        for i in range(100):
            # Codificar seed
            x_pred = np.zeros((1, seq_length, vocab_size))
            for t, char in enumerate(seed):
                x_pred[0, t, char_to_idx[char]] = 1

            # Prever próximo char
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample_with_temperature(preds, temperature)
            next_char = idx_to_char[next_index]

            generated += next_char
            seed = seed[1:] + next_char

        print(f"Generated: {generated}")

generate_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# ─── 7. TREINAR ───
print("\n🚀 Treinando modelo...")

history = model.fit(
    X, y,
    batch_size=128,
    epochs=5,  # Reduzido para demonstração
    callbacks=[generate_callback],
    verbose=1
)

# ─── 8. FUNÇÃO DE GERAÇÃO FINAL ───
def generate_text(seed_text, length=200, temperature=0.5):
    """
    Gera texto a partir de um seed
    """
    generated = seed_text
    seed = seed_text[-seq_length:]

    for i in range(length):
        x_pred = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(seed):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_index]

        generated += next_char
        seed = seed[1:] + next_char

    return generated

# ─── 9. GERAR EXEMPLOS FINAIS ───
print("\n" + "="*70)
print("📖 GERANDO TEXTO FINAL")
print("="*70)

seed_texts = ["to be or not to be", "the heart", "to die"]

for seed in seed_texts:
    # Pad seed se necessário
    if len(seed) < seq_length:
        seed = seed + " " * (seq_length - len(seed))

    print(f"\n🌱 Seed: '{seed[:30]}...'")

    for temp in [0.2, 0.7, 1.2]:
        generated = generate_text(seed, length=100, temperature=temp)
        print(f"\nTemp {temp}: {generated[:150]}...")

print("\n✅ GERAÇÃO DE TEXTO COMPLETA!")

print("\n💡 CONCEITOS:")
print("  • Character-level: Modelo prevê próximo caractere")
print("  • Temperature: Controla aleatoriedade (0=determinístico, 1+=criativo)")
print("  • Seq_length: Contexto usado para predição")
print("  • One-hot encoding: Representa caracteres como vetores")

print("\n📚 APLICAÇÕES:")
print("  • Autocompletar código")
print("  • Geração de poesia/histórias")
print("  • Chatbots criativos")
print("  • Composição musical (MIDI)")
