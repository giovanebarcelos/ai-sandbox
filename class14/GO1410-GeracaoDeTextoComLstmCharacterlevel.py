# GO1410-GeraçãoDeTextoComLstmCharacterlevel
# ═══════════════════════════════════════════════════════════════════
# GERAÇÃO DE TEXTO COM LSTM - CHARACTER-LEVEL
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import random
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# ─── 1. TEXTO DE EXEMPLO ───
# Pode usar Shakespeare, Machine Learning papers, etc.
text = """
Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. Deep learning is a 
subset of machine learning that uses neural networks with many layers. Natural language 
processing is another important area of AI that focuses on the interaction between 
computers and human language. Computer vision enables machines to interpret and understand 
visual information from the world.
""".lower()

print(f"Corpus length: {len(text)} caracteres")
print(f"Unique characters: {len(set(text))}")

# ─── 2. CRIAR VOCABULÁRIO ───
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

vocab_size = len(chars)
print(f"Vocabulário: {chars[:20]}...")

# ─── 3. CRIAR SEQUÊNCIAS ───
maxlen = 40  # Usar últimos 40 caracteres para predizer próximo
step = 3     # Passo da janela deslizante

sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print(f"Sequências criadas: {len(sentences)}")

# ─── 4. VETORIZAR ───
X = np.zeros((len(sentences), maxlen, vocab_size), dtype=bool)
y = np.zeros((len(sentences), vocab_size), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

print(f"X shape: {X.shape}")  # (num_sequences, 40, vocab_size)
print(f"y shape: {y.shape}")  # (num_sequences, vocab_size)

# ─── 5. CONSTRUIR MODELO ───
model_text = Sequential([
    LSTM(128, input_shape=(maxlen, vocab_size)),
    Dense(vocab_size, activation='softmax')
], name='text_generation_lstm')

model_text.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

model_text.summary()

# ─── 6. FUNÇÃO DE SAMPLING ───
def sample(preds, temperature=1.0):
    """
    Temperature sampling:
    - Low (0.5): mais conservador, repetitivo
    - Medium (1.0): balanceado
    - High (1.5): mais criativo, aleatório
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ─── 7. TREINAR E GERAR ───
print("\nTreinando modelo de geração de texto...")

for epoch in range(10):
    print(f"\nÉpoca {epoch+1}/10")
    model_text.fit(X, y, batch_size=128, epochs=1, verbose=0)

    # Gerar texto a cada época
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = text[start_index:start_index + maxlen]
    print(f"\nSeed: '{generated}'")

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f"\n---Temperature {temperature}---")
        sentence = generated
        print(sentence, end='')

        # Gerar 200 caracteres
        for _ in range(200):
            x_pred = np.zeros((1, maxlen, vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_idx[char]] = 1

            preds = model_text.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = idx_to_char[next_index]

            sentence = sentence[1:] + next_char
            print(next_char, end='')
        print()

# ═══════════════════════════════════════════════════════════════════
# OBSERVAÇÕES:
# • Temperature baixa (0.2): texto mais coerente mas repetitivo
# • Temperature alta (1.5): texto mais criativo mas incoerente
# • 1.0 é geralmente um bom balanço
# • Treinar por mais épocas melhora qualidade
# ═══════════════════════════════════════════════════════════════════
