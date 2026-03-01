# GO1438-AumentarDropout
# 1. Aumentar Dropout

if __name__ == "__main__":
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.3),  # Ao invés de 0.2
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(FORECAST_HORIZON)
    ])

    # 2. Usar Early Stopping (já implementado)
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

    # 3. Reduzir complexidade
    # LSTM(32) ao invés de LSTM(64)
