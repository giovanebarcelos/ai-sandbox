# GO1439-AumentarLookback
# 1. Aumentar lookback significativamente

if __name__ == "__main__":
    LOOKBACK = 72  # 3 dias de histórico

    # 2. Verificar se modelo tem capacidade suficiente
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(LOOKBACK, 1)),  # Mais neurônios
        Dropout(0.2),
        LSTM(64, return_sequences=True),  # Camada extra
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(FORECAST_HORIZON)
    ])

    # 3. Treinar mais épocas
    epochs = 200  # Ao invés de 100
