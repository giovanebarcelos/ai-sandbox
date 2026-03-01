# GO1405-AnáliseDeSentimentoComLstmImdb
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DE SENTIMENTO COM LSTM - IMDB MOVIE REVIEWS
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# ─── 1. CARREGAR DADOS ───
print("Carregando dataset IMDB...")
vocab_size = 10000  # Top 10k palavras mais frequentes
max_length = 200    # Truncar/pad para 200 palavras

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print(f"Train: {len(X_train)} reviews")
print(f"Test: {len(X_test)} reviews")
print(f"Exemplo review (IDs): {X_train[0][:20]}...")
print(f"Label: {y_train[0]} (0=negativo, 1=positivo)")

# ─── 2. PRÉ-PROCESSAMENTO ───
# Padding: todas reviews com mesmo tamanho
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

print(f"\nShape após padding:")
print(f"X_train: {X_train.shape}")  # (25000, 200)
print(f"X_test: {X_test.shape}")    # (25000, 200)

# ─── 3. CONSTRUIR MODELO LSTM ───
embedding_dim = 128
lstm_units = 64

model = Sequential([
    # Embedding layer: transforma IDs em vetores densos
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        name='embedding'
    ),

    # LSTM com dropout
    LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2, name='lstm'),

    # Camadas densas
    Dense(32, activation='relu', name='dense'),
    Dropout(0.5),

    # Output: probabilidade de ser positivo
    Dense(1, activation='sigmoid', name='output')
], name='sentiment_lstm')

model.summary()

# ─── 4. COMPILAR ───
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ─── 5. TREINAR ───
print("\nTreinando modelo...")

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# ─── 6. AVALIAR ───
print("\nAvaliando no conjunto de teste...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")  # ~87%

# ─── 7. VISUALIZAR TREINAMENTO ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy
ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ─── 8. TESTAR EM NOVOS TEXTOS ───
def predict_sentiment(text, model, word_index, max_length=200):
    """
    Prediz sentimento de uma review
    """
    # Tokenizar manualmente (simplificado)
    words = text.lower().split()
    sequence = [word_index.get(word, 0) for word in words]

    # Padding
    padded = pad_sequences([sequence], maxlen=max_length, padding='post')

    # Predição
    prediction = model.predict(padded, verbose=0)[0][0]

    sentiment = "POSITIVO" if prediction > 0.5 else "NEGATIVO"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return sentiment, confidence

# Obter word_index
word_index = imdb.get_word_index()

# Testes
test_reviews = [
    "This movie was absolutely amazing! Best film I've ever seen.",
    "Terrible waste of time. Boring and predictable.",
    "It was okay, nothing special but not bad either."
]

print("\n" + "="*60)
print("TESTANDO EM NOVOS TEXTOS")
print("="*60)

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review, model, word_index, max_length)
    print(f"\nReview: {review}")
    print(f"Sentimento: {sentiment} (confiança: {confidence*100:.1f}%)")
