# GO1418-36cOpcionalSentimentAnalysisComLstm
# ═══════════════════════════════════════════════════════════════════
# ANÁLISE DE SENTIMENTO COM LSTM BIDIRECIONAL
# Classificar reviews como positivas ou negativas
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

print("😊 SENTIMENT ANALYSIS COM BIDIRECTIONAL LSTM")
print("=" * 70)

# ─── 1. CARREGAR DADOS IMDB ───
print("\n📦 Carregando dataset IMDB...")

vocab_size = 10000
max_length = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print(f"  Train: {len(X_train)} reviews")
print(f"  Test: {len(X_test)} reviews")
print(f"  Exemplo review (word IDs): {X_train[0][:20]}...")
print(f"  Label: {y_train[0]} (0=negative, 1=positive)")

# ─── 2. PADDING ───
print("\n🔧 Aplicando padding...")

X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")

# ─── 3. MODELO UNIDIRECIONAL (BASELINE) ───
print("\n🏗️ Modelo 1: LSTM Unidirecional (baseline)...")

model_uni = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='UniLSTM')

model_uni.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"  Parâmetros: {model_uni.count_params():,}")

history_uni = model_uni.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,  # Reduzido para demonstração
    batch_size=128,
    verbose=1
)

test_loss_uni, test_acc_uni = model_uni.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {test_acc_uni:.4f}")

# ─── 4. MODELO BIDIRECIONAL ───
print("\n🏗️ Modelo 2: LSTM Bidirecional...")

model_bi = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
], name='BiLSTM')

model_bi.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"  Parâmetros: {model_bi.count_params():,}")

history_bi = model_bi.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3,
    batch_size=128,
    verbose=1
)

test_loss_bi, test_acc_bi = model_bi.evaluate(X_test, y_test, verbose=0)
print(f"  Test Accuracy: {test_acc_bi:.4f}")

# ─── 5. COMPARAR MODELOS ───
print("\n📊 Comparando modelos...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history_uni.history['accuracy'], label='Uni Train', linewidth=2)
axes[0].plot(history_uni.history['val_accuracy'], label='Uni Val', linewidth=2, linestyle='--')
axes[0].plot(history_bi.history['accuracy'], label='Bi Train', linewidth=2)
axes[0].plot(history_bi.history['val_accuracy'], label='Bi Val', linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history_uni.history['loss'], label='Uni Train', linewidth=2)
axes[1].plot(history_uni.history['val_loss'], label='Uni Val', linewidth=2, linestyle='--')
axes[1].plot(history_bi.history['loss'], label='Bi Train', linewidth=2)
axes[1].plot(history_bi.history['val_loss'], label='Bi Val', linewidth=2, linestyle='--')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Unidirectional vs Bidirectional LSTM', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('lstm_comparison.png', dpi=150)
print("✅ Comparação salva: lstm_comparison.png")

# ─── 6. TESTAR EM NOVOS TEXTOS ───
print("\n🧪 Testando em novos textos...")

def predict_sentiment(text, model, word_index, max_length=200):
    """
    Prediz sentimento de uma review
    """
    # Tokenizar simplificado
    words = text.lower().split()
    sequence = [word_index.get(word, 0) for word in words]

    # Padding
    padded = pad_sequences([sequence], maxlen=max_length, padding='post')

    # Predição
    prediction = model.predict(padded, verbose=0)[0][0]

    sentiment = "POSITIVO 😊" if prediction > 0.5 else "NEGATIVO 😞"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return sentiment, confidence, prediction

# Obter word_index
word_index = imdb.get_word_index()

# Testes
test_reviews = [
    "This movie was absolutely amazing! Best film I've ever seen.",
    "Terrible waste of time. Boring and predictable.",
    "It was okay, nothing special but not bad either.",
    "Brilliant acting and stunning visuals. Highly recommend!",
    "Worst movie ever. Don't waste your money."
]

print("\n" + "="*70)
print("TESTANDO EM NOVOS TEXTOS")
print("="*70)

for review in test_reviews:
    print(f"\nReview: \"{review}\"")

    # Unidirecional
    sent_uni, conf_uni, pred_uni = predict_sentiment(review, model_uni, word_index)
    print(f"  Uni-LSTM:  {sent_uni} (confidence: {conf_uni:.2%}, score: {pred_uni:.4f})")

    # Bidirecional
    sent_bi, conf_bi, pred_bi = predict_sentiment(review, model_bi, word_index)
    print(f"  Bi-LSTM:   {sent_bi} (confidence: {conf_bi:.2%}, score: {pred_bi:.4f})")

# ─── 7. ANÁLISE FINAL ───
print("\n" + "="*70)
print("📊 ANÁLISE FINAL")
print("="*70)

improvement = (test_acc_bi - test_acc_uni) / test_acc_uni * 100

print(f"\n🎯 Resultados:")
print(f"  Unidirecional:  {test_acc_uni:.4f}")
print(f"  Bidirecional:   {test_acc_bi:.4f}")
print(f"  Melhoria:       +{improvement:.2f}%")

print(f"\n💡 BIDIRECTIONAL LSTM:")
print(f"  • Processa sequência em ambas direções (forward + backward)")
print(f"  • Captura contexto passado E futuro")
print(f"  • Dobro de parâmetros vs unidirecional")
print(f"  • Melhor para tarefas onde contexto futuro importa")
print(f"  • Exemplo: 'not' em 'not good' → contexto futuro crucial")

print(f"\n📚 QUANDO USAR:")
print(f"  ✓ Sentiment analysis (contexto completo ajuda)")
print(f"  ✓ Named Entity Recognition")
print(f"  ✓ POS tagging")
print(f"  ✗ Real-time generation (não tem contexto futuro)")
print(f"  ✗ Streaming data")

print("\n✅ SENTIMENT ANALYSIS COMPLETA!")
