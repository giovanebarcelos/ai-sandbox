# GO1406-LstmBidirecionalContextoFuturoEPassado
# ═══════════════════════════════════════════════════════════════════
# LSTM BIDIRECIONAL - CONTEXTO FUTURO E PASSADO
# ═══════════════════════════════════════════════════════════════════

from tensorflow.keras.layers import Bidirectional

# ─── Modelo com Bidirectional LSTM ───


if __name__ == "__main__":
    model_bi = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),

        # Bidirectional LSTM: processa ambas direções
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),

        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name='bi_lstm_sentiment')

    model_bi.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Treinando Bidirectional LSTM...")
    history_bi = model_bi.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )

    # Avaliar
    test_loss_bi, test_acc_bi = model_bi.evaluate(X_test, y_test, verbose=0)
    print(f"\nBidirectional LSTM Test Accuracy: {test_acc_bi:.4f}")  # ~88-89%

    # ─── Comparação: LSTM vs Bidirectional ───
    print("\n" + "="*50)
    print("COMPARAÇÃO DE MODELOS")
    print("="*50)
    print(f"LSTM simples:       {test_accuracy:.4f}")
    print(f"Bidirectional LSTM: {test_acc_bi:.4f}")
    print(f"Melhoria: {(test_acc_bi - test_accuracy)*100:.2f}%")
